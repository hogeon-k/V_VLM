from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.defect_info import Detection
from service.onnx_detector import (
    DEFAULT_CLASS_NAMES,
    OnnxDetector,
    bbox_iou,
    detection_to_dict,
    summarize_timings,
)


PASS_LIMITS = {
    "pt_only": 0,
    "onnx_only": 0,
    "avg_conf_diff": 0.01,
    "max_conf_diff": 0.03,
    "avg_bbox_iou": 0.99,
    "min_bbox_iou": 0.95,
}


@dataclass(slots=True)
class DetectionMatch:
    status: str
    pt: Detection | None = None
    onnx: Detection | None = None
    iou: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Ultralytics PyTorch YOLO and ONNX Runtime outputs.")
    parser.add_argument("--image", type=Path, default=Path("data/images/01_short_01.jpg"))
    parser.add_argument("--pt-model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--onnx-model", type=Path, default=Path("models/best.onnx"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results/pytorch_onnx_compare"))
    return parser.parse_args()


def load_class_names(data_yaml: Path = Path("datasets/pcb/data.yaml")) -> dict[int, str]:
    if not data_yaml.exists():
        return DEFAULT_CLASS_NAMES
    try:
        import yaml

        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        names = data.get("names", {})
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, list):
            return {index: str(value) for index, value in enumerate(names)}
    except Exception:
        return DEFAULT_CLASS_NAMES
    return DEFAULT_CLASS_NAMES


def choose_image(path: Path) -> Path:
    if path.exists():
        return path
    candidates = sorted(Path("data/images").glob("*.jpg"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No image found at {path} or data/images/*.jpg")


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def run_pytorch_once(model: Any, image_path: Path, imgsz: int, conf: float, iou: float, device: str) -> tuple[list[Detection], dict[str, float]]:
    synchronize_cuda()
    start_total = time.perf_counter()
    prediction = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        batch=1,
        device=device,
        rect=False,
        save=False,
        verbose=False,
    )
    synchronize_cuda()
    total_ms = (time.perf_counter() - start_total) * 1000
    if not prediction:
        raise RuntimeError(f"PyTorch YOLO did not return a result for {image_path}")

    result = prediction[0]
    detections = pytorch_result_to_detections(result)
    speed = getattr(result, "speed", {}) or {}
    timings = {
        "preprocess_ms": float(speed.get("preprocess", 0.0)),
        "inference_ms": float(speed.get("inference", total_ms)),
        "postprocess_ms": float(speed.get("postprocess", 0.0)),
        "total_ms": float(total_ms),
    }
    return detections, timings


def pytorch_result_to_detections(result: Any) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = getattr(result, "names", {}) or {}
    detections: list[Detection] = []
    for xyxy, confidence, class_id_float in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), strict=True):
        class_id = int(class_id_float)
        x1, y1, x2, y2 = (int(round(float(value))) for value in xyxy)
        detections.append(
            Detection(
                class_id=class_id,
                class_name=str(names.get(class_id, DEFAULT_CLASS_NAMES.get(class_id, class_id))),
                confidence=float(confidence),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return detections


def measure_pytorch(model_path: Path, image_path: Path, imgsz: int, conf: float, iou: float, device: str, warmup: int, runs: int) -> tuple[list[Detection], dict[str, dict[str, float]]]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    detections: list[Detection] = []
    for _ in range(warmup):
        detections, _ = run_pytorch_once(model, image_path, imgsz, conf, iou, device)

    samples: list[dict[str, float]] = []
    for _ in range(runs):
        detections, timings = run_pytorch_once(model, image_path, imgsz, conf, iou, device)
        samples.append(timings)
    return detections, summarize_timings(samples)


def measure_onnx(detector: OnnxDetector, image_path: Path, warmup: int, runs: int) -> tuple[Any, dict[str, dict[str, float]]]:
    timed = None
    for _ in range(warmup):
        timed = detector.detect_timed(image_path)

    samples: list[dict[str, float]] = []
    for _ in range(runs):
        timed = detector.detect_timed(image_path)
        samples.append(
            {
                "preprocess_ms": timed.preprocess_ms,
                "inference_ms": timed.inference_ms,
                "postprocess_ms": timed.postprocess_ms,
                "total_ms": timed.total_ms,
            }
        )
    if timed is None:
        timed = detector.detect_timed(image_path)
    return timed, summarize_timings(samples)


def match_detections(pt: list[Detection], onnx: list[Detection], match_iou: float = 0.5) -> list[DetectionMatch]:
    matches: list[DetectionMatch] = []
    used_onnx: set[int] = set()
    for pt_detection in pt:
        best_index: int | None = None
        best_iou = 0.0
        for index, onnx_detection in enumerate(onnx):
            if index in used_onnx or pt_detection.class_id != onnx_detection.class_id:
                continue
            current_iou = bbox_iou(_bbox(pt_detection), _bbox(onnx_detection))
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = index
        if best_index is not None and best_iou >= match_iou:
            used_onnx.add(best_index)
            matches.append(DetectionMatch(status="MATCHED", pt=pt_detection, onnx=onnx[best_index], iou=best_iou))
        else:
            matches.append(DetectionMatch(status="PT_ONLY", pt=pt_detection))

    for index, onnx_detection in enumerate(onnx):
        if index not in used_onnx:
            matches.append(DetectionMatch(status="ONNX_ONLY", onnx=onnx_detection))
    return matches


def _bbox(detection: Detection) -> list[int]:
    return [detection.x1, detection.y1, detection.x2, detection.y2]


def summarize_accuracy(matches: list[DetectionMatch]) -> tuple[dict[str, Any], list[str]]:
    matched = [match for match in matches if match.status == "MATCHED" and match.pt is not None and match.onnx is not None]
    pt_only = [match for match in matches if match.status == "PT_ONLY"]
    onnx_only = [match for match in matches if match.status == "ONNX_ONLY"]
    conf_diffs = [abs(float(match.pt.confidence) - float(match.onnx.confidence)) for match in matched]
    bbox_ious = [match.iou for match in matched]

    summary = {
        "matched_count": len(matched),
        "pt_only_count": len(pt_only),
        "onnx_only_count": len(onnx_only),
        "classes_match": all(match.pt.class_id == match.onnx.class_id for match in matched),
        "avg_conf_diff": float(np.mean(conf_diffs)) if conf_diffs else 0.0,
        "max_conf_diff": float(np.max(conf_diffs)) if conf_diffs else 0.0,
        "avg_bbox_iou": float(np.mean(bbox_ious)) if bbox_ious else 0.0,
        "min_bbox_iou": float(np.min(bbox_ious)) if bbox_ious else 0.0,
    }

    warnings: list[str] = []
    if summary["pt_only_count"] > PASS_LIMITS["pt_only"]:
        warnings.append(f"PT_ONLY count {summary['pt_only_count']} > {PASS_LIMITS['pt_only']}")
    if summary["onnx_only_count"] > PASS_LIMITS["onnx_only"]:
        warnings.append(f"ONNX_ONLY count {summary['onnx_only_count']} > {PASS_LIMITS['onnx_only']}")
    if not summary["classes_match"]:
        warnings.append("Matched class ids are not identical")
    if matched and summary["avg_conf_diff"] > PASS_LIMITS["avg_conf_diff"]:
        warnings.append(f"Average confidence diff {summary['avg_conf_diff']:.6f} > {PASS_LIMITS['avg_conf_diff']}")
    if matched and summary["max_conf_diff"] > PASS_LIMITS["max_conf_diff"]:
        warnings.append(f"Max confidence diff {summary['max_conf_diff']:.6f} > {PASS_LIMITS['max_conf_diff']}")
    if matched and summary["avg_bbox_iou"] < PASS_LIMITS["avg_bbox_iou"]:
        warnings.append(f"Average bbox IoU {summary['avg_bbox_iou']:.6f} < {PASS_LIMITS['avg_bbox_iou']}")
    if matched and summary["min_bbox_iou"] < PASS_LIMITS["min_bbox_iou"]:
        warnings.append(f"Min bbox IoU {summary['min_bbox_iou']:.6f} < {PASS_LIMITS['min_bbox_iou']}")
    return summary, warnings


def match_to_dict(match: DetectionMatch) -> dict[str, Any]:
    row: dict[str, Any] = {"status": match.status, "bbox_iou": float(match.iou)}
    if match.pt is not None:
        row["pytorch"] = detection_to_dict(match.pt)
    if match.onnx is not None:
        row["onnx"] = detection_to_dict(match.onnx)
    if match.pt is not None and match.onnx is not None:
        row["class_id"] = match.pt.class_id
        row["class_name"] = match.pt.class_name
        row["confidence_diff_abs"] = abs(float(match.pt.confidence) - float(match.onnx.confidence))
        row["bbox_diff_abs"] = [abs(a - b) for a, b in zip(_bbox(match.pt), _bbox(match.onnx), strict=True)]
    return row


def write_csv(matches: list[DetectionMatch], output_path: Path) -> None:
    fields = [
        "status",
        "class_id",
        "class_name",
        "pt_confidence",
        "onnx_confidence",
        "confidence_diff_abs",
        "pt_bbox",
        "onnx_bbox",
        "bbox_iou",
        "bbox_diff_abs",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for match in matches:
            pt = match.pt
            onnx = match.onnx
            writer.writerow(
                {
                    "status": match.status,
                    "class_id": pt.class_id if pt is not None else (onnx.class_id if onnx is not None else ""),
                    "class_name": pt.class_name if pt is not None else (onnx.class_name if onnx is not None else ""),
                    "pt_confidence": pt.confidence if pt is not None else "",
                    "onnx_confidence": onnx.confidence if onnx is not None else "",
                    "confidence_diff_abs": abs(pt.confidence - onnx.confidence) if pt is not None and onnx is not None else "",
                    "pt_bbox": _bbox(pt) if pt is not None else "",
                    "onnx_bbox": _bbox(onnx) if onnx is not None else "",
                    "bbox_iou": match.iou,
                    "bbox_diff_abs": [abs(a - b) for a, b in zip(_bbox(pt), _bbox(onnx), strict=True)] if pt is not None and onnx is not None else "",
                }
            )


def draw_detections(image_path: Path, detections: list[Detection], output_path: Path, title: str, class_names: dict[int, str]) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Input image not found or unreadable: {image_path}")
    colors = {
        0: (40, 180, 255),
        1: (80, 220, 90),
        2: (230, 90, 120),
    }
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(image, f"{title} | detections: {len(detections)}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    for detection in detections:
        color = colors.get(detection.class_id, (255, 255, 255))
        cv2.rectangle(image, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
        label = f"{class_names.get(detection.class_id, detection.class_name)} {detection.confidence:.3f}"
        y = max(16, detection.y1 - 6)
        cv2.putText(image, label, (detection.x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write image: {output_path}")


def write_side_by_side(left_path: Path, right_path: Path, output_path: Path) -> None:
    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))
    if left is None or right is None:
        raise FileNotFoundError("Annotated image missing for side-by-side output")
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
    combined = np.hstack([left, right])
    if not cv2.imwrite(str(output_path), combined):
        raise RuntimeError(f"Failed to write image: {output_path}")


def build_report(
    args: argparse.Namespace,
    image_path: Path,
    class_names: dict[int, str],
    pt_detections: list[Detection],
    onnx_timed: Any,
    pt_timing: dict[str, dict[str, float]],
    onnx_timing: dict[str, dict[str, float]],
    matches: list[DetectionMatch],
    accuracy: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    speedup = 0.0
    if onnx_timing["total_ms"]["avg"] > 0:
        speedup = pt_timing["total_ms"]["avg"] / onnx_timing["total_ms"]["avg"]
    return {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "models": {
            "pytorch": str(args.pt_model),
            "onnx": str(args.onnx_model),
        },
        "image": {
            "path": str(image_path),
            "shape": list(cv2.imread(str(image_path)).shape[:2]),
        },
        "input_settings": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "batch": 1,
            "device": args.device,
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "onnx_runtime": {
            "providers": onnx_timed.providers,
            "input_name": onnx_timed.input_name,
            "output_name": onnx_timed.output_name,
            "input_shape": onnx_timed.input_shape,
            "output_shape": onnx_timed.output_shape,
        },
        "detections": {
            "pytorch": [detection_to_dict(detection) for detection in pt_detections],
            "onnx": [detection_to_dict(detection) for detection in onnx_timed.detections],
        },
        "matches": [match_to_dict(match) for match in matches],
        "accuracy_summary": accuracy,
        "timing": {
            "pytorch": pt_timing,
            "onnx": onnx_timing,
            "speedup_pt_over_onnx_total": speedup,
        },
        "judgement": {
            "status": "PASS" if not warnings else "WARNING",
            "warnings": warnings,
            "limits": PASS_LIMITS,
        },
        "notes": [
            "PyTorch stage timings use Ultralytics result.speed values; total_ms is measured wall time around model.predict.",
            "ONNX Runtime CUDA timing is measured around session.run; provider-level GPU synchronization is not exposed directly.",
        ],
    }


def print_summary(report: dict[str, Any]) -> None:
    accuracy = report["accuracy_summary"]
    pt_avg = report["timing"]["pytorch"]["total_ms"]["avg"]
    onnx_avg = report["timing"]["onnx"]["total_ms"]["avg"]
    speedup = report["timing"]["speedup_pt_over_onnx_total"]
    print("=== PyTorch vs ONNX Runtime comparison ===")
    print(f"PyTorch model: {report['models']['pytorch']}")
    print(f"ONNX model:    {report['models']['onnx']}")
    print(f"Image:         {report['image']['path']}")
    print(f"Image shape:   {report['image']['shape']} (height, width)")
    print(f"imgsz/conf/iou: {report['input_settings']['imgsz']} / {report['input_settings']['conf']} / {report['input_settings']['iou']}")
    print(f"PyTorch detections: {len(report['detections']['pytorch'])}")
    print(f"ONNX detections:    {len(report['detections']['onnx'])}")
    print(f"Matched/PT only/ONNX only: {accuracy['matched_count']} / {accuracy['pt_only_count']} / {accuracy['onnx_only_count']}")
    print(f"Classes match: {accuracy['classes_match']}")
    print(f"Avg/max confidence diff: {accuracy['avg_conf_diff']:.6f} / {accuracy['max_conf_diff']:.6f}")
    print(f"Avg/min matched bbox IoU: {accuracy['avg_bbox_iou']:.6f} / {accuracy['min_bbox_iou']:.6f}")
    print(f"PyTorch avg total time: {pt_avg:.3f} ms")
    print(f"ONNX avg total time:    {onnx_avg:.3f} ms")
    print(f"Speedup PT/ONNX:        {speedup:.3f}x")
    print(f"Final judgement: {report['judgement']['status']}")
    for warning in report["judgement"]["warnings"]:
        print(f"WARNING: {warning}")


def main() -> int:
    args = parse_args()
    image_path = choose_image(args.image)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names()

    pt_detections, pt_timing = measure_pytorch(args.pt_model, image_path, args.imgsz, args.conf, args.iou, str(args.device), args.warmup, args.runs)
    onnx_detector = OnnxDetector(args.onnx_model, imgsz=args.imgsz, conf=args.conf, iou=args.iou, class_names=class_names)
    onnx_timed, onnx_timing = measure_onnx(onnx_detector, image_path, args.warmup, args.runs)

    matches = match_detections(pt_detections, onnx_timed.detections)
    accuracy, warnings = summarize_accuracy(matches)
    report = build_report(args, image_path, class_names, pt_detections, onnx_timed, pt_timing, onnx_timing, matches, accuracy, warnings)

    json_path = args.output_dir / "comparison.json"
    csv_path = args.output_dir / "comparison.csv"
    pt_image_path = args.output_dir / "pytorch_result.jpg"
    onnx_image_path = args.output_dir / "onnx_result.jpg"
    side_by_side_path = args.output_dir / "side_by_side.jpg"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(matches, csv_path)
    draw_detections(image_path, pt_detections, pt_image_path, "PyTorch", class_names)
    draw_detections(image_path, onnx_timed.detections, onnx_image_path, "ONNX Runtime", class_names)
    write_side_by_side(pt_image_path, onnx_image_path, side_by_side_path)
    print_summary(report)
    print(f"Results written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
