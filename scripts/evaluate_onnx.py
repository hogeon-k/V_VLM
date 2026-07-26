from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import compare_predictions as gt_eval
from model.defect_info import Detection as AppDetection
from scripts.compare_pytorch_onnx import load_class_names as load_class_name_map
from scripts.compare_pytorch_onnx import match_detections as match_backend_detections
from scripts.compare_pytorch_onnx_batch import TimingSample, collect_images, summarize_backend_timings
from service.onnx_detector import (
    OnnxDetector,
    bbox_iou,
    class_aware_nms,
    detection_to_dict,
    preprocess_image,
    restore_boxes_to_original,
    validate_onnx_output,
    xywh_to_xyxy,
)


CSV_ENCODING = "utf-8-sig"
AP_IOU_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.5, 1.0, 0.05))


@dataclass(slots=True)
class PredictionBundle:
    predictions: dict[str, list[gt_eval.Detection]]
    app_predictions: dict[str, list[AppDetection]]
    timings: dict[str, list[TimingSample]]
    runtime: dict[str, Any] = field(default_factory=dict)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ONNX YOLO detections on a labelled YOLO dataset split.")
    parser.add_argument("--model", type=Path, default=Path("models/best.onnx"))
    parser.add_argument("--pytorch-model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("datasets/pcb/data.yaml"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--match-iou", type=float, default=0.5, help="Ground-truth matching IoU threshold.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/onnx/evaluation"))
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--map-iou", type=float, nargs="*", default=list(AP_IOU_THRESHOLDS))
    parser.add_argument("--fail-map50-diff", type=float, default=0.01)
    parser.add_argument("--fail-map5095-diff", type=float, default=0.01)
    parser.add_argument("--fail-precision-diff", type=float, default=0.02)
    parser.add_argument("--fail-recall-diff", type=float, default=0.02)
    parser.add_argument("--min-avg-matched-box-iou", type=float, default=0.99)
    parser.add_argument("--max-new-fp", type=int, default=0)
    parser.add_argument("--max-new-fn", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Write detailed ONNX postprocess trace files.")
    parser.add_argument("--debug-image", type=Path, help="Image path or filename to trace through ONNX postprocess.")
    parser.add_argument("--save-raw-output", action="store_true", help="Save raw ONNX Runtime output as raw_output.npy during debug.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if ONNX Runtime falls back from CUDAExecutionProvider.")
    return parser.parse_args(argv)


def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_dataset_paths(data_yaml: Path, split: str) -> tuple[Path, Path, list[str]]:
    data_yaml = resolve_project_path(data_yaml)
    class_names = gt_eval.load_class_names(data_yaml)
    config = gt_eval.yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if split not in config:
        raise ValueError(f"data.yaml does not define split '{split}'.")
    base = Path(str(config.get("path", data_yaml.parent))).expanduser()
    if not base.is_absolute():
        project_relative = (PROJECT_ROOT / base).resolve()
        yaml_relative = (data_yaml.parent / base).resolve()
        base = project_relative if project_relative.exists() or not yaml_relative.exists() else yaml_relative
    split_value = Path(str(config[split])).expanduser()
    image_dir = split_value if split_value.is_absolute() else (base / split_value).resolve()
    parts = list(image_dir.parts)
    if "images" not in parts:
        raise ValueError(f"Image split path must contain an 'images' directory so labels can be resolved: {image_dir}")
    parts[parts.index("images")] = "labels"
    label_dir = Path(*parts)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image split directory does not exist: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label split directory does not exist: {label_dir}")
    return image_dir, label_dir, class_names


def to_gt_detection(detection: AppDetection) -> gt_eval.Detection:
    return gt_eval.Detection(
        class_id=int(detection.class_id),
        box=(float(detection.x1), float(detection.y1), float(detection.x2), float(detection.y2)),
        confidence=float(detection.confidence),
    )


def to_app_detection(detection: gt_eval.Detection, class_names: list[str]) -> AppDetection:
    x1, y1, x2, y2 = detection.box
    return AppDetection(
        class_id=detection.class_id,
        class_name=gt_eval.safe_name(class_names, detection.class_id),
        confidence=float(detection.confidence or 0.0),
        x1=int(round(x1)),
        y1=int(round(y1)),
        x2=int(round(x2)),
        y2=int(round(y2)),
    )


def load_ground_truth(images: list[Path], label_dir: Path, class_names: list[str]) -> dict[str, list[gt_eval.Detection]]:
    ground_truth: dict[str, list[gt_eval.Detection]] = {}
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Input image is unreadable: {image_path}")
        height, width = image.shape[:2]
        ground_truth[image_path.name] = gt_eval.read_yolo_labels(label_dir / f"{image_path.stem}.txt", width, height, len(class_names))
    return ground_truth


def run_onnx_predictions(args: argparse.Namespace, images: list[Path], class_names: list[str]) -> PredictionBundle:
    detector = OnnxDetector(
        resolve_project_path(args.model),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        class_names={i: name for i, name in enumerate(class_names)},
        require_cuda=args.require_cuda,
    )
    predictions: dict[str, list[gt_eval.Detection]] = {}
    app_predictions: dict[str, list[AppDetection]] = {}
    timings: dict[str, list[TimingSample]] = {}
    runtime: dict[str, Any] = {}
    for image_path in images:
        timed, float_detections = detect_onnx_float(detector, image_path, class_names)
        predictions[image_path.name] = float_detections
        app_predictions[image_path.name] = [to_app_detection(detection, class_names) for detection in float_detections]
        timings[image_path.name] = [
            TimingSample(
                "onnx",
                0,
                timed.preprocess_ms,
                timed.inference_ms,
                timed.postprocess_ms,
                timed.total_ms,
                timed.providers[0] if timed.providers else "",
            )
        ]
        runtime = {
            "providers": timed.providers,
            "input_name": timed.input_name,
            "output_name": timed.output_name,
            "input_shape": timed.input_shape,
            "output_shape": timed.output_shape,
        }
        if args.require_cuda and "CUDAExecutionProvider" not in timed.providers:
            raise RuntimeError(
                "CUDAExecutionProvider was required, but ONNX Runtime used "
                f"{timed.providers}. Run scripts/diagnose_onnxruntime_cuda.py for details."
            )
    return PredictionBundle(predictions, app_predictions, timings, runtime)


def detect_onnx_float(detector: OnnxDetector, image_path: Path, class_names: list[str]) -> tuple[Any, list[gt_eval.Detection]]:
    timed = detector.detect_timed(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Input image is unreadable: {image_path}")
    session = detector._load_session()
    input_tensor, letterbox_info = preprocess_image(image, detector.imgsz)
    raw_output = np.asarray(session.run([detector._output_name], {detector._input_name: input_tensor})[0])
    rows = validate_onnx_output(raw_output)
    boxes = restore_boxes_to_original(xywh_to_xyxy(rows[:, :4]), letterbox_info)
    class_scores = rows[:, 4:]
    class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
    confidences = np.max(class_scores, axis=1).astype(np.float32)
    candidates = confidences >= detector.conf
    if not np.any(candidates):
        return timed, []
    boxes = boxes[candidates]
    scores = confidences[candidates]
    classes = class_ids[candidates]
    keep = class_aware_nms(boxes, scores, classes, detector.iou)
    detections = [
        gt_eval.Detection(
            class_id=int(classes[index]),
            box=tuple(float(value) for value in boxes[index]),
            confidence=float(scores[index]),
        )
        for index in keep
    ]
    return timed, detections


def run_pytorch_predictions(args: argparse.Namespace, images: list[Path], class_names: list[str]) -> PredictionBundle:
    from ultralytics import YOLO
    try:
        import torch
    except Exception:
        torch = None

    model = YOLO(str(resolve_project_path(args.pytorch_model)))
    predictions: dict[str, list[gt_eval.Detection]] = {}
    app_predictions: dict[str, list[AppDetection]] = {}
    timings: dict[str, list[TimingSample]] = {}
    for image_path in images:
        start = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            batch=1,
            device=args.device,
            rect=False,
            save=False,
            verbose=False,
        )[0]
        total_ms = (time.perf_counter() - start) * 1000
        detections = gt_eval.result_to_detections(result)
        predictions[image_path.name] = detections
        app_predictions[image_path.name] = [to_app_detection(detection, class_names) for detection in detections]
        speed = getattr(result, "speed", {}) or {}
        timings[image_path.name] = [
            TimingSample(
                "pytorch",
                0,
                float(speed.get("preprocess", 0.0)),
                float(speed.get("inference", total_ms)),
                float(speed.get("postprocess", 0.0)),
                total_ms,
                str(args.device),
            )
        ]
    requested_device = str(args.device)
    actual_device = "cuda" if requested_device != "cpu" and torch is not None and torch.cuda.is_available() else "cpu"
    return PredictionBundle(predictions, app_predictions, timings, {"requested_device": requested_device, "actual_device": actual_device})


def evaluate_predictions(
    model_name: str,
    predictions_by_image: dict[str, list[gt_eval.Detection]],
    ground_truth_by_image: dict[str, list[gt_eval.Detection]],
    class_names: list[str],
    match_iou: float,
    ap_ious: tuple[float, ...],
    timings_by_image: dict[str, list[TimingSample]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = gt_eval.initialise_counts(class_names)
    per_image: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    for image_name, gt in ground_truth_by_image.items():
        predictions = predictions_by_image.get(image_name, [])
        result = gt_eval.analyse_image(gt, predictions, class_names, match_iou)
        for class_name, counts in result["per_class"].items():
            for key, value in counts.items():
                totals[class_name][key] += value
        error_records.extend(gt_eval.build_error_records(model_name, image_name, gt, result, class_names, 0.0, match_iou, False))
        per_image.append(
            {
                "image_name": image_name,
                "gt_count": len(gt),
                "prediction_count": len(predictions),
                "tp": result["tp"],
                "fp": result["fp"],
                "fn": result["fn"],
                "timing": summarize_backend_timings(timings_by_image.get(image_name, [])),
            }
        )
    per_class_ap = compute_map(predictions_by_image, ground_truth_by_image, class_names, ap_ious)
    per_class = {}
    overall_counts = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
    for class_name in class_names:
        row = gt_eval.metric_row(totals[class_name])
        ap_row = per_class_ap[class_name]
        per_class[class_name] = {**row, "AP50": ap_row.get(0.5), "AP50-95": mean_or_zero(list(ap_row.values()))}
        for key in overall_counts:
            overall_counts[key] += int(totals[class_name][key])
    overall_row = gt_eval.metric_row(overall_counts)
    metrics = {
        "model": model_name,
        "overall": {
            "precision": overall_row["precision"],
            "recall": overall_row["recall"],
            "f1": overall_row["f1"],
            "mAP50": mean_or_zero([per_class[name]["AP50"] for name in class_names]),
            "mAP50-95": mean_or_zero([per_class[name]["AP50-95"] for name in class_names]),
            "tp": overall_counts["tp"],
            "fp": overall_counts["fp"],
            "fn": overall_counts["fn"],
            "gt": overall_counts["gt"],
            "pred": overall_counts["pred"],
        },
        "per_class": per_class,
        "per_image": per_image,
        "timing_summary": summarize_backend_timings([sample for samples in timings_by_image.values() for sample in samples]),
        "map_iou_thresholds": list(ap_ious),
        "matching": {
            "nms_iou_is_separate_from_match_iou": True,
            "match_iou": match_iou,
            "method": "confidence-descending one-to-one same-class greedy matching; AP uses 101-point interpolated precision envelope.",
        },
    }
    return metrics, error_records


def compute_map(
    predictions_by_image: dict[str, list[gt_eval.Detection]],
    ground_truth_by_image: dict[str, list[gt_eval.Detection]],
    class_names: list[str],
    iou_thresholds: tuple[float, ...],
) -> dict[str, dict[float, float]]:
    result: dict[str, dict[float, float]] = {name: {} for name in class_names}
    for class_id, class_name in enumerate(class_names):
        gt_by_image = {
            image_name: [gt for gt in gts if gt.class_id == class_id]
            for image_name, gts in ground_truth_by_image.items()
        }
        preds = [
            (image_name, pred)
            for image_name, predictions in predictions_by_image.items()
            for pred in predictions
            if pred.class_id == class_id
        ]
        preds.sort(key=lambda item: float(item[1].confidence or 0.0), reverse=True)
        gt_count = sum(len(items) for items in gt_by_image.values())
        for threshold in iou_thresholds:
            result[class_name][float(threshold)] = ap_for_threshold(preds, gt_by_image, gt_count, float(threshold))
    return result


def ap_for_threshold(
    preds: list[tuple[str, gt_eval.Detection]],
    gt_by_image: dict[str, list[gt_eval.Detection]],
    gt_count: int,
    threshold: float,
) -> float:
    if gt_count == 0:
        return 0.0
    used: dict[str, set[int]] = {image_name: set() for image_name in gt_by_image}
    tp: list[float] = []
    fp: list[float] = []
    for image_name, pred in preds:
        best_index = -1
        best_iou = 0.0
        for index, gt in enumerate(gt_by_image.get(image_name, [])):
            if index in used[image_name]:
                continue
            iou = gt_eval.box_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= threshold:
            used[image_name].add(best_index)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not tp:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(gt_count, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return interpolated_ap(recall, precision)


def interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    recall_points = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(recall_points, mrec, mpre), recall_points))


def compare_backends(
    pytorch_metrics: dict[str, Any],
    onnx_metrics: dict[str, Any],
    pytorch_predictions: dict[str, list[gt_eval.Detection]],
    onnx_predictions: dict[str, list[gt_eval.Detection]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overall_pt = pytorch_metrics["overall"]
    overall_onnx = onnx_metrics["overall"]
    differences = {
        "precision": overall_onnx["precision"] - overall_pt["precision"],
        "recall": overall_onnx["recall"] - overall_pt["recall"],
        "f1": overall_onnx["f1"] - overall_pt["f1"],
        "mAP50": overall_onnx["mAP50"] - overall_pt["mAP50"],
        "mAP50-95": overall_onnx["mAP50-95"] - overall_pt["mAP50-95"],
        "tp": overall_onnx["tp"] - overall_pt["tp"],
        "fp": overall_onnx["fp"] - overall_pt["fp"],
        "fn": overall_onnx["fn"] - overall_pt["fn"],
    }
    rows: list[dict[str, Any]] = []
    class_mismatch_count = 0
    matched_ious: list[float] = []
    conf_diffs: list[float] = []
    for image_name in sorted(set(pytorch_predictions) | set(onnx_predictions)):
        matches = match_float_predictions(pytorch_predictions.get(image_name, []), onnx_predictions.get(image_name, []), args.match_iou)
        matched = [match for match in matches if match["status"] == "MATCHED"]
        image_ious = [float(match["iou"]) for match in matched]
        image_conf_diffs = [float(match["confidence_diff_abs"]) for match in matched]
        matched_ious.extend(image_ious)
        conf_diffs.extend(image_conf_diffs)
        rows.append(
            {
                "image_name": image_name,
                "pytorch_count": len(pytorch_predictions.get(image_name, [])),
                "onnx_count": len(onnx_predictions.get(image_name, [])),
                "matched": len(matched),
                "pytorch_only": sum(match["status"] == "PT_ONLY" for match in matches),
                "onnx_only": sum(match["status"] == "ONNX_ONLY" for match in matches),
                "avg_confidence_diff": mean_or_zero(image_conf_diffs),
                "avg_bbox_iou": mean_or_zero(image_ious),
            }
        )
    avg_iou = mean_or_zero(matched_ious)
    new_fp = max(0, int(differences["fp"]))
    new_fn = max(0, int(differences["fn"]))
    warnings = []
    if abs(differences["mAP50"]) > args.fail_map50_diff:
        warnings.append("mAP50 difference exceeds threshold")
    if abs(differences["mAP50-95"]) > args.fail_map5095_diff:
        warnings.append("mAP50-95 difference exceeds threshold")
    if abs(differences["precision"]) > args.fail_precision_diff:
        warnings.append("precision difference exceeds threshold")
    if abs(differences["recall"]) > args.fail_recall_diff:
        warnings.append("recall difference exceeds threshold")
    if class_mismatch_count:
        warnings.append("class mismatch count is not zero")
    if new_fp > args.max_new_fp:
        warnings.append("new FP count exceeds threshold")
    if new_fn > args.max_new_fn:
        warnings.append("new FN count exceeds threshold")
    if matched_ious and avg_iou < args.min_avg_matched_box_iou:
        warnings.append("average matched box IoU is below threshold")
    status = "PASS" if not warnings else "WARNING"
    pytorch_device = pytorch_metrics.get("runtime", {}).get("actual_device")
    onnx_provider = onnx_metrics.get("runtime", {}).get("providers", [""])[0]
    speed_valid, speed_reason = speed_comparison_validity(pytorch_device, onnx_provider)
    comparison = {
        "overall_result": status,
        "differences": differences,
        "class_ap_differences": {
            class_name: {
                "AP50": onnx_metrics["per_class"][class_name]["AP50"] - pytorch_metrics["per_class"][class_name]["AP50"],
                "AP50-95": onnx_metrics["per_class"][class_name]["AP50-95"] - pytorch_metrics["per_class"][class_name]["AP50-95"],
            }
            for class_name in onnx_metrics["per_class"]
        },
        "average_confidence_difference": mean_or_zero(conf_diffs),
        "average_matched_box_iou": avg_iou,
        "class_mismatch_count": class_mismatch_count,
        "new_fp_count": new_fp,
        "new_fn_count": new_fn,
        "bbox_evaluation_dtype": "float64",
        "display_bbox_dtype": "int",
        "accuracy_comparison_valid": True,
        "thresholds": {
            "mAP50_difference": args.fail_map50_diff,
            "mAP50_95_difference": args.fail_map5095_diff,
            "precision_difference": args.fail_precision_diff,
            "recall_difference": args.fail_recall_diff,
            "new_fp_count": args.max_new_fp,
            "new_fn_count": args.max_new_fn,
            "average_matched_box_iou": args.min_avg_matched_box_iou,
        },
        "warnings": warnings,
        "speed_comparison_valid": speed_valid,
        "speed_comparison_reason": speed_reason,
        "notes": [
            "PASS/WARNING compares metric deltas against configurable thresholds.",
            "FAIL is reserved for fatal execution/model errors that prevent evaluation.",
        ],
    }
    return comparison, rows


def match_float_predictions(
    pytorch: list[gt_eval.Detection],
    onnx: list[gt_eval.Detection],
    match_iou: float,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    used_onnx: set[int] = set()
    for pt_detection in pytorch:
        best_index: int | None = None
        best_iou = 0.0
        for index, onnx_detection in enumerate(onnx):
            if index in used_onnx or pt_detection.class_id != onnx_detection.class_id:
                continue
            current_iou = gt_eval.box_iou(pt_detection, onnx_detection)
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = index
        if best_index is not None and best_iou >= match_iou:
            used_onnx.add(best_index)
            onnx_detection = onnx[best_index]
            matches.append(
                {
                    "status": "MATCHED",
                    "iou": best_iou,
                    "confidence_diff_abs": abs(float(pt_detection.confidence or 0.0) - float(onnx_detection.confidence or 0.0)),
                    "pytorch": pt_detection,
                    "onnx": onnx_detection,
                }
            )
        else:
            matches.append({"status": "PT_ONLY", "iou": 0.0, "confidence_diff_abs": 0.0, "pytorch": pt_detection, "onnx": None})
    for index, onnx_detection in enumerate(onnx):
        if index not in used_onnx:
            matches.append({"status": "ONNX_ONLY", "iou": 0.0, "confidence_diff_abs": 0.0, "pytorch": None, "onnx": onnx_detection})
    return matches


def speed_comparison_validity(pytorch_device: str | None, onnx_provider: str | None) -> tuple[bool, str]:
    if not pytorch_device or not onnx_provider:
        return False, "Runtime device/provider information is incomplete."
    pytorch_uses_cuda = str(pytorch_device).startswith("cuda")
    onnx_uses_cuda = onnx_provider == "CUDAExecutionProvider"
    if pytorch_uses_cuda == onnx_uses_cuda:
        return True, "PyTorch and ONNX Runtime used comparable accelerator classes."
    return False, f"PyTorch used {pytorch_device} while ONNX Runtime used {onnx_provider}."


def analyse_new_fn_cases(
    ground_truth_by_image: dict[str, list[gt_eval.Detection]],
    pytorch_predictions: dict[str, list[gt_eval.Detection]],
    onnx_predictions: dict[str, list[gt_eval.Detection]],
    class_names: list[str],
    match_iou: float,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for image_name, gt in ground_truth_by_image.items():
        pt_result = gt_eval.analyse_image(gt, pytorch_predictions.get(image_name, []), class_names, match_iou)
        onnx_result = gt_eval.analyse_image(gt, onnx_predictions.get(image_name, []), class_names, match_iou)
        pt_matched_gt = {match.gt_index: match for match in pt_result["tp_matches"]}
        onnx_matched_gt = {match.gt_index for match in onnx_result["tp_matches"]}
        for gt_index, pt_match in pt_matched_gt.items():
            if gt_index in onnx_matched_gt:
                continue
            gt_detection = gt[gt_index]
            pt_prediction = pytorch_predictions[image_name][pt_match.pred_index]
            onnx_same = best_candidate_for_gt(gt_detection, onnx_predictions.get(image_name, []), same_class=True)
            onnx_any = best_candidate_for_gt(gt_detection, onnx_predictions.get(image_name, []), same_class=False)
            cases.append(
                {
                    "image_name": image_name,
                    "gt_index": gt_index,
                    "gt": gt_detection_to_dict(gt_detection, class_names),
                    "pytorch_match": {
                        "prediction": gt_detection_to_dict(pt_prediction, class_names),
                        "gt_iou": pt_match.iou,
                    },
                    "onnx_prediction_count": len(onnx_predictions.get(image_name, [])),
                    "onnx_best_same_class_candidate": candidate_summary(onnx_same, gt_detection, class_names),
                    "onnx_best_any_class_candidate": candidate_summary(onnx_any, gt_detection, class_names),
                    "reason": classify_fn_reason(gt_detection, onnx_same, onnx_any, match_iou),
                }
            )
    return {
        "new_fn_count": len(cases),
        "match_iou": match_iou,
        "cases": cases,
    }


def classify_fn_reason(
    gt_detection: gt_eval.Detection,
    same_class_candidate: gt_eval.Detection | None,
    any_class_candidate: gt_eval.Detection | None,
    match_iou: float,
) -> str:
    if same_class_candidate is not None and gt_eval.box_iou(gt_detection, same_class_candidate) < match_iou:
        return "match_iou_below_threshold_after_final_detections"
    if any_class_candidate is not None and any_class_candidate.class_id != gt_detection.class_id and gt_eval.box_iou(gt_detection, any_class_candidate) >= match_iou:
        return "class_mismatch_after_final_detections"
    return "no_same_class_final_detection_near_gt"


def best_candidate_for_gt(
    gt_detection: gt_eval.Detection,
    predictions: list[gt_eval.Detection],
    same_class: bool,
) -> gt_eval.Detection | None:
    candidates = [
        prediction
        for prediction in predictions
        if not same_class or prediction.class_id == gt_detection.class_id
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda prediction: gt_eval.box_iou(gt_detection, prediction))


def candidate_summary(candidate: gt_eval.Detection | None, gt_detection: gt_eval.Detection, class_names: list[str]) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        **gt_detection_to_dict(candidate, class_names),
        "gt_iou": gt_eval.box_iou(gt_detection, candidate),
    }


def gt_detection_to_dict(detection: gt_eval.Detection, class_names: list[str]) -> dict[str, Any]:
    return {
        "class_id": int(detection.class_id),
        "class_name": gt_eval.safe_name(class_names, detection.class_id),
        "confidence": detection.confidence,
        "bbox": [float(value) for value in detection.box],
    }


def select_debug_image(args: argparse.Namespace, images: list[Path], fn_analysis: dict[str, Any]) -> Path | None:
    if args.debug_image is not None:
        requested = resolve_project_path(args.debug_image)
        if requested.is_file():
            return requested
        for image_path in images:
            if image_path.name == args.debug_image.name:
                return image_path
        raise FileNotFoundError(f"--debug-image does not match an image file: {args.debug_image}")
    if args.debug and fn_analysis.get("cases"):
        target_name = fn_analysis["cases"][0]["image_name"]
        return next((image_path for image_path in images if image_path.name == target_name), None)
    if args.debug and images:
        return images[0]
    return None


def write_onnx_debug_trace(
    args: argparse.Namespace,
    image_path: Path,
    label_dir: Path,
    class_names: list[str],
    pytorch_predictions: dict[str, list[gt_eval.Detection]],
    onnx_predictions: dict[str, list[gt_eval.Detection]],
    fn_analysis: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Debug image is unreadable: {image_path}")

    detector = OnnxDetector(
        resolve_project_path(args.model),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        class_names={i: name for i, name in enumerate(class_names)},
        require_cuda=args.require_cuda,
    )
    session = detector._load_session()
    input_tensor, letterbox_info = preprocess_image(image, args.imgsz)
    raw_output = np.asarray(session.run([detector._output_name], {detector._input_name: input_tensor})[0])
    if args.save_raw_output:
        np.save(debug_dir / "raw_output.npy", raw_output)

    decoded = decode_onnx_output(raw_output, class_names)
    after_conf = enrich_debug_candidates([item for item in decoded if item["confidence"] >= args.conf], letterbox_info)
    after_nms = apply_debug_nms(after_conf, letterbox_info, args.iou)
    final_detections = [gt_detection_to_dict(detection, class_names) for detection in onnx_predictions.get(image_path.name, [])]

    write_json(debug_dir / "decoded_candidates.json", {"count": len(decoded), "candidates": decoded})
    write_json(debug_dir / "after_conf_filter.json", {"conf": args.conf, "count": len(after_conf), "candidates": after_conf})
    write_json(debug_dir / "after_nms.json", {"iou": args.iou, "count": len(after_nms), "candidates": after_nms})
    write_json(debug_dir / "final_detections.json", {"count": len(final_detections), "detections": final_detections})

    gt = gt_eval.read_yolo_labels(label_dir / f"{image_path.stem}.txt", image.shape[1], image.shape[0], len(class_names))
    image_cases = [case for case in fn_analysis.get("cases", []) if case["image_name"] == image_path.name]
    debug_summary = {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "settings": {"imgsz": args.imgsz, "conf": args.conf, "nms_iou": args.iou, "match_iou": args.match_iou},
        "preprocess": {
            "opencv_imread_bgr": True,
            "bgr_to_rgb": True,
            "letterbox": {
                "original_shape": list(letterbox_info.original_shape),
                "resized_shape": list(letterbox_info.resized_shape),
                "ratio": list(letterbox_info.ratio),
                "pad_left_top": list(letterbox_info.pad),
                "padding_value": 114,
                "interpolation": "cv2.INTER_LINEAR",
            },
            "tensor": {
                "layout": "NCHW",
                "dtype": str(input_tensor.dtype),
                "normalized_by_255": True,
                "contiguous": bool(input_tensor.flags["C_CONTIGUOUS"]),
                "shape": list(input_tensor.shape),
            },
        },
        "output_interpretation": {
            "raw_shape": list(raw_output.shape),
            "transpose": "[1, 7, N] -> [N, 7]",
            "box_format": "xywh in letterboxed input coordinates",
            "class_score_channels": len(class_names),
            "objectness_channel": False,
            "extra_sigmoid_applied": False,
            "already_nms": False,
        },
        "counts": {
            "decoded_candidates": len(decoded),
            "after_conf_filter": len(after_conf),
            "after_nms": len(after_nms),
            "final_detections": len(final_detections),
        },
        "new_fn_cases_for_image": image_cases,
        "stage_reason": classify_debug_stage(image_cases, after_conf, after_nms, gt, class_names, args.match_iou),
    }
    write_json(debug_dir / "debug_summary.json", debug_summary)
    draw_debug_comparison(debug_dir / "debug_comparison.png", image_path, gt, pytorch_predictions.get(image_path.name, []), onnx_predictions.get(image_path.name, []), class_names)
    return debug_summary


def decode_onnx_output(raw_output: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    rows = validate_onnx_output(raw_output)
    boxes_xywh = rows[:, :4]
    class_scores = rows[:, 4:]
    class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
    confidences = np.max(class_scores, axis=1).astype(np.float32)
    decoded = []
    for index, (box, class_id, confidence, scores) in enumerate(zip(boxes_xywh, class_ids, confidences, class_scores, strict=True)):
        decoded.append(
            {
                "index": index,
                "bbox_xywh_letterbox": [float(value) for value in box],
                "class_id": int(class_id),
                "class_name": gt_eval.safe_name(class_names, int(class_id)),
                "confidence": float(confidence),
                "class_scores": [float(value) for value in scores],
            }
        )
    return decoded


def apply_debug_nms(candidates: list[dict[str, Any]], letterbox_info: Any, iou: float) -> list[dict[str, Any]]:
    if not candidates:
        return []
    boxes_xywh = np.asarray([candidate["bbox_xywh_letterbox"] for candidate in candidates], dtype=np.float32)
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    boxes_original = restore_boxes_to_original(boxes_xyxy, letterbox_info)
    scores = np.asarray([candidate["confidence"] for candidate in candidates], dtype=np.float32)
    classes = np.asarray([candidate["class_id"] for candidate in candidates], dtype=np.int32)
    keep = class_aware_nms(boxes_original, scores, classes, iou)
    kept = []
    for index in keep:
        item = dict(candidates[index])
        item["bbox_xyxy_original"] = [float(value) for value in boxes_original[index]]
        kept.append(item)
    return kept


def enrich_debug_candidates(candidates: list[dict[str, Any]], letterbox_info: Any) -> list[dict[str, Any]]:
    if not candidates:
        return []
    boxes_xywh = np.asarray([candidate["bbox_xywh_letterbox"] for candidate in candidates], dtype=np.float32)
    boxes_original = restore_boxes_to_original(xywh_to_xyxy(boxes_xywh), letterbox_info)
    enriched = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        item["bbox_xyxy_original"] = [float(value) for value in boxes_original[index]]
        enriched.append(item)
    return enriched


def classify_debug_stage(
    image_cases: list[dict[str, Any]],
    after_conf: list[dict[str, Any]],
    after_nms: list[dict[str, Any]],
    gt: list[gt_eval.Detection],
    class_names: list[str],
    match_iou: float,
) -> dict[str, Any]:
    if not image_cases:
        return {"classification": "no_new_fn_for_debug_image"}
    case = image_cases[0]
    gt_box = case["gt"]["bbox"]
    gt_class_id = int(case["gt"]["class_id"])
    conf_best = best_debug_candidate(gt_box, gt_class_id, after_conf, class_names, same_class=True)
    nms_best = best_debug_candidate(gt_box, gt_class_id, after_nms, class_names, same_class=True)
    if conf_best is None:
        classification = "removed_by_confidence_filter_or_absent_in_raw_output"
    elif nms_best is None:
        classification = "removed_by_nms"
    elif nms_best["gt_iou"] < match_iou:
        classification = "kept_after_nms_but_match_iou_below_threshold"
    elif image_cases[0].get("onnx_best_same_class_candidate") and image_cases[0]["onnx_best_same_class_candidate"].get("gt_iou", 0.0) < match_iou:
        classification = "coordinate_rounding_after_nms_dropped_iou_below_match_threshold"
    else:
        classification = "kept_after_nms_but_not_matched_likely_one_to_one_assignment"
    return {
        "classification": classification,
        "best_after_conf_filter": conf_best,
        "best_after_nms": nms_best,
    }


def best_debug_candidate(
    gt_box: list[float],
    gt_class_id: int,
    candidates: list[dict[str, Any]],
    class_names: list[str],
    same_class: bool,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_iou = -1.0
    for candidate in candidates:
        if same_class and candidate["class_id"] != gt_class_id:
            continue
        bbox = candidate.get("bbox_xyxy_original")
        if bbox is None:
            continue
        current = bbox_iou(gt_box, bbox)
        if current > best_iou:
            best_iou = current
            best = {**candidate, "gt_iou": current}
    return best


def draw_debug_comparison(
    path: Path,
    image_path: Path,
    gt: list[gt_eval.Detection],
    pytorch_predictions: list[gt_eval.Detection],
    onnx_predictions: list[gt_eval.Detection],
    class_names: list[str],
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    panels = [
        gt_eval.add_header(gt_eval.annotate_ground_truth(image, gt, class_names), "Ground truth"),
        gt_eval.add_header(gt_eval.annotate_predictions(image, pytorch_predictions, gt, *gt_eval.match_detections(gt, pytorch_predictions, 0.5)[:2], class_names), "PyTorch"),
        gt_eval.add_header(gt_eval.annotate_predictions(image, onnx_predictions, gt, *gt_eval.match_detections(gt, onnx_predictions, 0.5)[:2], class_names), "ONNX Runtime"),
    ]
    target_height = min(800, max(panel.shape[0] for panel in panels))
    combined = cv2.hconcat([gt_eval.resize_panel(panel, target_height) for panel in panels])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), combined)


def write_predictions(path: Path, predictions: dict[str, list[gt_eval.Detection]], class_names: list[str]) -> None:
    payload = {
        image_name: [prediction_record(detection, class_names) for detection in detections]
        for image_name, detections in sorted(predictions.items())
    }
    write_json(path, payload)


def prediction_record(detection: gt_eval.Detection, class_names: list[str]) -> dict[str, Any]:
    display = tuple(int(round(value)) for value in detection.box)
    return {
        "class_id": int(detection.class_id),
        "class_name": gt_eval.safe_name(class_names, detection.class_id),
        "confidence": float(detection.confidence or 0.0),
        "evaluation_bbox": [float(value) for value in detection.box],
        "display_bbox": list(display),
        "bbox": [float(value) for value in detection.box],
    }


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["image_name", "pytorch_count", "onnx_count", "matched", "pytorch_only", "onnx_only", "avg_confidence_diff", "avg_bbox_iou"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_failure_cases(output_dir: Path, error_records: list[dict[str, Any]]) -> None:
    failure_dir = output_dir / "failure_cases"
    failure_dir.mkdir(parents=True, exist_ok=True)
    selected = [record for record in error_records if record["error_type"] in {"FP", "FN"}]
    write_json(failure_dir / "failure_cases.json", selected)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    return value


def mean_or_zero(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def write_diagnostic_report(
    path: Path,
    final_payload: dict[str, Any],
    fn_analysis: dict[str, Any],
    onnx_metrics: dict[str, Any],
    pytorch_metrics: dict[str, Any],
    debug_summary: dict[str, Any] | None,
) -> None:
    cases = fn_analysis.get("cases") or []
    case = cases[0] if cases else None
    lines = ["# G1 ONNX Diagnostic Report", "", "## 1. New FN Target Image"]
    if case is None:
        lines.append("No GT-level new FN was found relative to PyTorch.")
    else:
        lines.extend(
            [
                f"- Image: `{case['image_name']}`",
                f"- Reason: `{case['reason']}`",
                "",
                "## 2. GT Information",
                f"- Class: `{case['gt']['class_name']}`",
                f"- BBox: `{case['gt']['bbox']}`",
                "",
                "## 3. PyTorch Result",
                f"- Class: `{case['pytorch_match']['prediction']['class_name']}`",
                f"- Confidence: `{case['pytorch_match']['prediction']['confidence']}`",
                f"- BBox: `{case['pytorch_match']['prediction']['bbox']}`",
                f"- IoU with GT: `{case['pytorch_match']['gt_iou']}`",
                "",
                "## 4. ONNX Result",
                f"- ONNX final detection count: `{case['onnx_prediction_count']}`",
                f"- Best same-class final candidate: `{case['onnx_best_same_class_candidate']}`",
                f"- Best any-class final candidate: `{case['onnx_best_any_class_candidate']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 5. Stage Trace",
            f"- Debug summary: `{debug_summary.get('stage_reason') if debug_summary else 'not generated'}`",
            "",
            "## 6. FN Cause",
            "- The diagnostic classification is based on raw decode, confidence filter, class-aware NMS, and final GT matching when debug output is generated.",
            "",
            "## 7. Accuracy Impact",
            f"- Precision diff: `{final_payload.get('differences', {}).get('precision')}`",
            f"- Recall diff: `{final_payload.get('differences', {}).get('recall')}`",
            f"- mAP50 diff: `{final_payload.get('differences', {}).get('mAP50')}`",
            f"- mAP50-95 diff: `{final_payload.get('differences', {}).get('mAP50-95')}`",
            "",
            "## 8. Fix Required",
            "- No threshold or model-output correction was applied in this diagnostic run.",
            "- Recommended evaluation fix: keep restored ONNX boxes as float coordinates for metric matching, then round only for GUI drawing/storage.",
            "",
            "## 9. CUDA Provider Failure Cause",
            "- See `cuda_diagnostics.json` for package, provider, PATH, DLL, and session creation details.",
            "",
            "## 10. Recommended Environment Fix",
            "- Align ONNX Runtime GPU requirements with the installed CUDA/cuDNN/MSVC runtime, or run both PyTorch and ONNX on CPU for fair speed checks.",
            "",
            "## 11. Speed Comparison Validity",
            f"- Valid: `{final_payload.get('speed_comparison_valid')}`",
            f"- Reason: `{final_payload.get('speed_comparison_reason')}`",
            "",
            "## 12. G1 Final Judgement",
            f"- Result: `{g1_judgement(final_payload, fn_analysis)}`",
            f"- Original comparison result: `{final_payload.get('overall_result')}`",
            "",
            "## Metric Snapshot",
            f"- PyTorch overall: `{pytorch_metrics.get('overall')}`",
            f"- ONNX overall: `{onnx_metrics.get('overall')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def g1_judgement(final_payload: dict[str, Any], fn_analysis: dict[str, Any]) -> str:
    if final_payload.get("overall_result") == "FAIL":
        return "FAIL"
    if fn_analysis.get("new_fn_count", 0):
        diffs = final_payload.get("differences", {})
        thresholds = final_payload.get("thresholds", {})
        metric_within_limits = (
            abs(float(diffs.get("mAP50", 0.0))) <= float(thresholds.get("mAP50_difference", 0.0))
            and abs(float(diffs.get("mAP50-95", 0.0))) <= float(thresholds.get("mAP50_95_difference", 0.0))
            and abs(float(diffs.get("precision", 0.0))) <= float(thresholds.get("precision_difference", 0.0))
            and abs(float(diffs.get("recall", 0.0))) <= float(thresholds.get("recall_difference", 0.0))
        )
        return "CONDITIONAL PASS" if metric_within_limits else "WARNING"
    return "PASS" if final_payload.get("overall_result") == "PASS" else "WARNING"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve_project_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        image_dir, label_dir, class_names = resolve_dataset_paths(args.data, args.split)
        images = collect_images(image_dir, recursive=False)
        if not images:
            raise FileNotFoundError(f"No images found in split directory: {image_dir}")
        ground_truth = load_ground_truth(images, label_dir, class_names)
        ap_ious = tuple(float(value) for value in args.map_iou)

        onnx_bundle = run_onnx_predictions(args, images, class_names)
        onnx_metrics, onnx_errors = evaluate_predictions("onnx", onnx_bundle.predictions, ground_truth, class_names, args.match_iou, ap_ious, onnx_bundle.timings)
        onnx_metrics["runtime"] = onnx_bundle.runtime
        write_json(output_dir / "onnx_metrics.json", onnx_metrics)
        write_predictions(output_dir / "onnx_predictions.json", onnx_bundle.predictions, class_names)
        write_failure_cases(output_dir, onnx_errors)

        final_payload: dict[str, Any] = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "data": str(resolve_project_path(args.data)),
            "split": args.split,
            "onnx_metrics": onnx_metrics["overall"],
            "overall_result": "PASS",
        }
        if not args.skip_pytorch:
            pytorch_bundle = run_pytorch_predictions(args, images, class_names)
            pytorch_metrics, _pytorch_errors = evaluate_predictions("pytorch", pytorch_bundle.predictions, ground_truth, class_names, args.match_iou, ap_ious, pytorch_bundle.timings)
            pytorch_metrics["runtime"] = pytorch_bundle.runtime
            write_json(output_dir / "pytorch_metrics.json", pytorch_metrics)
            write_predictions(output_dir / "pytorch_predictions.json", pytorch_bundle.predictions, class_names)
            comparison, rows = compare_backends(pytorch_metrics, onnx_metrics, pytorch_bundle.predictions, onnx_bundle.predictions, args)
            final_payload.update(comparison)
            write_comparison_csv(output_dir / "pytorch_vs_onnx.csv", rows)
            fn_analysis = analyse_new_fn_cases(ground_truth, pytorch_bundle.predictions, onnx_bundle.predictions, class_names, args.match_iou)
            write_json(output_dir / "fn_analysis.json", fn_analysis)
            debug_summary = None
            debug_image = select_debug_image(args, images, fn_analysis)
            if args.debug or debug_image is not None:
                if debug_image is None:
                    raise FileNotFoundError("Debug was requested, but no debug image could be selected.")
                debug_summary = write_onnx_debug_trace(
                    args,
                    debug_image,
                    label_dir,
                    class_names,
                    pytorch_bundle.predictions,
                    onnx_bundle.predictions,
                    fn_analysis,
                    output_dir,
                )
            write_diagnostic_report(output_dir / "g1_diagnostic_report.md", final_payload, fn_analysis, onnx_metrics, pytorch_metrics, debug_summary)
        elif args.debug:
            raise ValueError("--debug requires PyTorch comparison; remove --skip-pytorch.")
        write_json(output_dir / "final_comparison.json", final_payload)
        print("=== ONNX Evaluation ===")
        print(f"Images: {len(images)}")
        print(f"Precision/Recall/F1: {onnx_metrics['overall']['precision']:.4f} / {onnx_metrics['overall']['recall']:.4f} / {onnx_metrics['overall']['f1']:.4f}")
        print(f"mAP50/mAP50-95: {onnx_metrics['overall']['mAP50']:.4f} / {onnx_metrics['overall']['mAP50-95']:.4f}")
        print(f"Overall result: {final_payload.get('overall_result', 'PASS')}")
        print(f"Output: {output_dir}")
        return 0
    except Exception as exc:
        failure = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "overall_result": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json(output_dir / "final_comparison.json", failure)
        print(f"ERROR: {failure['error']}", file=sys.stderr)
        print("Overall result: FAIL", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
