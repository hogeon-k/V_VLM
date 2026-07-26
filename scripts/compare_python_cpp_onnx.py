from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python ONNX and C++ ONNX single-image result JSON files.")
    parser.add_argument("--python-result", type=Path, default=Path("benchmarks/cpp_onnx/reference/python_onnx_result.json"))
    parser.add_argument("--cpp-result", type=Path, default=Path("benchmarks/cpp_onnx/single/result.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/cpp_onnx/comparison"))
    parser.add_argument("--match-iou", type=float, default=0.5)
    return parser.parse_args(argv)


def bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - inter
    return 0.0 if union <= 0 else float(inter / union)


def match_detections(python_detections: list[dict[str, Any]], cpp_detections: list[dict[str, Any]], match_iou: float) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    used_cpp: set[int] = set()
    for py_index, py_det in enumerate(python_detections):
        best_index: int | None = None
        best_iou = 0.0
        for cpp_index, cpp_det in enumerate(cpp_detections):
            if cpp_index in used_cpp or int(py_det["class_id"]) != int(cpp_det["class_id"]):
                continue
            current_iou = bbox_iou(py_det["bbox"], cpp_det["bbox"])
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = cpp_index
        if best_index is not None and best_iou >= match_iou:
            used_cpp.add(best_index)
            cpp_det = cpp_detections[best_index]
            matches.append(
                {
                    "status": "MATCHED",
                    "python_index": py_index,
                    "cpp_index": best_index,
                    "class_id": int(py_det["class_id"]),
                    "confidence_diff_abs": abs(float(py_det["confidence"]) - float(cpp_det["confidence"])),
                    "bbox_iou": best_iou,
                    "python": py_det,
                    "cpp": cpp_det,
                }
            )
        else:
            matches.append({"status": "PYTHON_ONLY", "python_index": py_index, "python": py_det, "bbox_iou": best_iou})

    for cpp_index, cpp_det in enumerate(cpp_detections):
        if cpp_index not in used_cpp:
            matches.append({"status": "CPP_ONLY", "cpp_index": cpp_index, "cpp": cpp_det, "bbox_iou": 0.0})
    return matches


def write_csv(path: Path, matches: list[dict[str, Any]]) -> None:
    fields = ["status", "class_id", "confidence_diff_abs", "bbox_iou", "python_bbox", "cpp_bbox"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for match in matches:
            writer.writerow(
                {
                    "status": match["status"],
                    "class_id": match.get("class_id", ""),
                    "confidence_diff_abs": match.get("confidence_diff_abs", ""),
                    "bbox_iou": match.get("bbox_iou", ""),
                    "python_bbox": match.get("python", {}).get("bbox", ""),
                    "cpp_bbox": match.get("cpp", {}).get("bbox", ""),
                }
            )


def write_side_by_side(path: Path, python_result: dict[str, Any], cpp_result: dict[str, Any]) -> None:
    image_path = python_result.get("image") or cpp_result.get("image")
    image = cv2.imread(str(image_path))
    if image is None:
        return

    def draw(detections: list[dict[str, Any]], title: str) -> np.ndarray:
        canvas = image.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(canvas, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        for det in detections:
            bbox = [int(round(float(value))) for value in det["bbox"]]
            cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (40, 180, 255), 2)
            cv2.putText(canvas, f"{det['class_name']} {float(det['confidence']):.3f}", (bbox[0], max(16, bbox[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 180, 255), 2, cv2.LINE_AA)
        return canvas

    combined = np.hstack([draw(python_result["detections"], "Python ONNX"), draw(cpp_result["detections"], "C++ ONNX")])
    cv2.imwrite(str(path), combined)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    python_result = json.loads(args.python_result.read_text(encoding="utf-8"))
    cpp_result = json.loads(args.cpp_result.read_text(encoding="utf-8"))
    matches = match_detections(python_result["detections"], cpp_result["detections"], args.match_iou)
    matched = [match for match in matches if match["status"] == "MATCHED"]
    python_only = [match for match in matches if match["status"] == "PYTHON_ONLY"]
    cpp_only = [match for match in matches if match["status"] == "CPP_ONLY"]
    conf_diffs = [float(match["confidence_diff_abs"]) for match in matched]
    bbox_ious = [float(match["bbox_iou"]) for match in matched]
    class_mismatch = 0

    summary = {
        "status": "PASS",
        "python_detection_count": len(python_result["detections"]),
        "cpp_detection_count": len(cpp_result["detections"]),
        "matched_count": len(matched),
        "python_only_count": len(python_only),
        "cpp_only_count": len(cpp_only),
        "class_mismatch_count": class_mismatch,
        "confidence_diff_max": max(conf_diffs) if conf_diffs else 0.0,
        "bbox_iou_min": min(bbox_ious) if bbox_ious else 1.0,
        "thresholds": {"confidence_diff_max": 0.01, "bbox_iou_min": 0.99, "match_iou": args.match_iou},
        "matches": matches,
    }
    if python_only or cpp_only or class_mismatch or summary["confidence_diff_max"] > 0.01 or summary["bbox_iou_min"] < 0.99:
        summary["status"] = "FAIL"

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output / "comparison.csv", matches)
    write_side_by_side(args.output / "side_by_side.jpg", python_result, cpp_result)
    print(f"Python/C++ ONNX comparison: {summary['status']}")
    print(f"Matched/Python only/C++ only: {len(matched)} / {len(python_only)} / {len(cpp_only)}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
