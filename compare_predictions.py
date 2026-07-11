from __future__ import annotations

"""Compare two YOLO models against the same labelled PCB test images.

The script writes prediction labels, annotated images, side-by-side images, and
per-image/class CSV reports. It deliberately performs matching itself so TP,
FP, FN, and wrong-class detections are evaluated with one documented rule.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

try:
    import yaml
except ImportError:  # Importing YOLO is delayed so --help always works.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_A = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_MODEL_B = PROJECT_ROOT / "runs" / "detect" / "pcb_ablation_scale05" / "weights" / "best.pt"
DEFAULT_IMAGES = PROJECT_ROOT / "datasets" / "pcb" / "images" / "test"
DEFAULT_LABELS = PROJECT_ROOT / "datasets" / "pcb" / "labels" / "test"
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "pcb" / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "prediction_compare"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# BGR colours used by OpenCV: green=ground truth, blue=TP, red=FP,
# orange=wrong-class prediction, yellow=unmatched ground-truth/FN.
GT_COLOR = (0, 200, 0)
TP_COLOR = (255, 130, 0)
FP_COLOR = (0, 0, 230)
CONFUSION_COLOR = (0, 145, 255)
FN_COLOR = (0, 220, 255)


@dataclass(frozen=True)
class Detection:
    """One bounding-box detection in pixel xyxy coordinates."""

    class_id: int
    box: tuple[float, float, float, float]
    confidence: float | None = None


@dataclass(frozen=True)
class Match:
    """A matched prediction/ground-truth pair, optionally a class confusion."""

    gt_index: int
    pred_index: int
    iou: float
    is_correct_class: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line options with immediately usable PCB defaults."""
    parser = argparse.ArgumentParser(
        description="Compare two YOLO models on the same labelled PCB test images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-a", default=str(DEFAULT_MODEL_A), help="First model best.pt path.")
    parser.add_argument("--model-b", default=str(DEFAULT_MODEL_B), help="Second model best.pt path.")
    parser.add_argument("--name-a", default="existing_best", help="Display/output name for model A.")
    parser.add_argument("--name-b", default="scale05", help="Display/output name for model B.")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES), help="Test image directory.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="YOLO ground-truth label directory.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML path for class names.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.15, help="Prediction confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for TP/FP/FN matching.")
    parser.add_argument("--device", default="0", help="YOLO device, for example 0 or cpu.")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="Base output directory.")
    parser.add_argument("--run-name", default=None, help="Optional execution subdirectory name.")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    """Resolve relative paths from the directory containing this script."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def list_best_pt_candidates() -> list[Path]:
    """Return available best.pt files useful when a supplied model path is invalid."""
    candidates = [PROJECT_ROOT / "models" / "best.pt"]
    candidates.extend((PROJECT_ROOT / "runs" / "detect").glob("*/weights/best.pt"))
    return sorted(path.resolve() for path in candidates if path.is_file())


def print_model_candidates() -> None:
    """Print local best.pt candidates without selecting one automatically."""
    print("Available best.pt candidates:")
    candidates = list_best_pt_candidates()
    if not candidates:
        print("  (none found)")
        return
    for path in candidates:
        try:
            print(f"  - {path.relative_to(PROJECT_ROOT)}")
        except ValueError:
            print(f"  - {path}")


def load_class_names(data_path: Path) -> list[str]:
    """Read class names from data.yaml, accepting list and index-mapping forms."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to read data.yaml. Install requirements.txt first.")
    with data_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    names = config.get("names") if isinstance(config, dict) else None
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        try:
            return [str(names[key]) for key in sorted(names, key=lambda key: int(key))]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid indexed 'names' mapping in {data_path}") from error
    raise ValueError(f"data.yaml must define 'names' as a list or an indexed mapping: {data_path}")


def validate_args(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve and validate all input paths and numeric settings before inference."""
    paths = {
        "model_a": resolve_path(args.model_a),
        "model_b": resolve_path(args.model_b),
        "images": resolve_path(args.images),
        "labels": resolve_path(args.labels),
        "data": resolve_path(args.data),
        "project": resolve_path(args.project),
    }
    missing_files = [key for key in ("model_a", "model_b", "data") if not paths[key].is_file()]
    if missing_files:
        for key in missing_files:
            print(f"[ERROR] {key.replace('_', '-')} file does not exist: {paths[key]}")
        print_model_candidates()
        raise FileNotFoundError("Required model or data.yaml file is missing.")
    for key in ("images", "labels"):
        if not paths[key].is_dir():
            raise FileNotFoundError(f"[ERROR] {key} directory does not exist: {paths[key]}")
    if args.imgsz <= 0:
        raise ValueError("[ERROR] --imgsz must be greater than 0.")
    for option in ("conf", "iou", "match_iou"):
        value = getattr(args, option)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"[ERROR] --{option} must be between 0 and 1.")
    if not args.name_a.strip() or not args.name_b.strip():
        raise ValueError("[ERROR] --name-a and --name-b must not be empty.")
    return paths


def find_images(images_dir: Path) -> list[Path]:
    """Find supported image files in deterministic filename order."""
    images = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"[ERROR] No jpg/jpeg/png/bmp images found in: {images_dir}")
    return images


def create_output_dir(project_dir: Path, requested_name: str | None) -> Path:
    """Create a non-destructive execution directory under the comparison project."""
    project_dir.mkdir(parents=True, exist_ok=True)
    base_name = requested_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = project_dir / base_name
    suffix = 1
    while candidate.exists():
        candidate = project_dir / f"{base_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def read_yolo_labels(label_path: Path, image_width: int, image_height: int, class_count: int) -> list[Detection]:
    """Read normalised YOLO labels and convert them exactly to pixel xyxy boxes."""
    if not label_path.is_file():
        print(f"[WARNING] Missing label file; treating image as having no ground truth: {label_path.name}")
        return []
    detections: list[Detection] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            print(f"[WARNING] Invalid label ignored ({label_path.name}:{line_number}): expected 5 values.")
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            print(f"[WARNING] Invalid numeric label ignored ({label_path.name}:{line_number}).")
            continue
        if not 0 <= class_id < class_count:
            print(f"[WARNING] Unknown class ID {class_id} ignored ({label_path.name}:{line_number}).")
            continue
        x1 = max(0.0, min(float(image_width), (cx - width / 2.0) * image_width))
        y1 = max(0.0, min(float(image_height), (cy - height / 2.0) * image_height))
        x2 = max(0.0, min(float(image_width), (cx + width / 2.0) * image_width))
        y2 = max(0.0, min(float(image_height), (cy + height / 2.0) * image_height))
        if x2 <= x1 or y2 <= y1:
            print(f"[WARNING] Empty label box ignored ({label_path.name}:{line_number}).")
            continue
        detections.append(Detection(class_id=class_id, box=(x1, y1, x2, y2)))
    return detections


def box_iou(first: Detection, second: Detection) -> float:
    """Calculate IoU for two pixel-coordinate xyxy bounding boxes."""
    ax1, ay1, ax2, ay2 = first.box
    bx1, by1, bx2, by2 = second.box
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def greedy_pairs(gt: list[Detection], predictions: list[Detection], match_iou: float, same_class: bool, used_gt: set[int], used_pred: set[int]) -> list[Match]:
    """Return descending-IoU one-to-one greedy pairs from unused detections."""
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt_detection in enumerate(gt):
        if gt_index in used_gt:
            continue
        for pred_index, prediction in enumerate(predictions):
            if pred_index in used_pred:
                continue
            if same_class and gt_detection.class_id != prediction.class_id:
                continue
            if not same_class and gt_detection.class_id == prediction.class_id:
                continue
            iou = box_iou(gt_detection, prediction)
            if iou >= match_iou:
                candidates.append((iou, gt_index, pred_index))
    matches: list[Match] = []
    for iou, gt_index, pred_index in sorted(candidates, key=lambda item: item[0], reverse=True):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append(Match(gt_index, pred_index, iou, same_class))
    return matches


def match_detections(gt: list[Detection], predictions: list[Detection], match_iou: float) -> tuple[list[Match], list[Match], set[int], set[int]]:
    """Match correct classes first, then record remaining wrong-class overlaps."""
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    true_positives = greedy_pairs(gt, predictions, match_iou, True, used_gt, used_pred)
    confusions = greedy_pairs(gt, predictions, match_iou, False, used_gt, used_pred)
    confusions = [match for match in confusions if gt[match.gt_index].class_id != predictions[match.pred_index].class_id]
    return true_positives, confusions, used_gt, used_pred


def result_to_detections(result: Any) -> list[Detection]:
    """Convert an Ultralytics Result boxes object to lightweight detections."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy_values = boxes.xyxy.detach().cpu().numpy()
    class_values = boxes.cls.detach().cpu().numpy().astype(int)
    confidence_values = boxes.conf.detach().cpu().numpy()
    detections: list[Detection] = []
    for coords, class_id, confidence in zip(xyxy_values, class_values, confidence_values, strict=True):
        detections.append(
            Detection(
                class_id=int(class_id),
                box=tuple(float(value) for value in coords),
                confidence=float(confidence),
            )
        )
    return detections


def safe_name(class_names: list[str], class_id: int) -> str:
    """Return a readable class name even for unexpected prediction IDs."""
    return class_names[class_id] if 0 <= class_id < len(class_names) else f"unknown_{class_id}"


def normalise_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    """Convert a pixel xyxy box to YOLO normalised centre-width-height format."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def save_prediction_labels(path: Path, predictions: list[Detection], width: int, height: int) -> None:
    """Save predictions as YOLO labels with confidence appended as a sixth field."""
    lines = []
    for prediction in predictions:
        cx, cy, box_width, box_height = normalise_box(prediction.box, width, height)
        confidence = prediction.confidence if prediction.confidence is not None else 0.0
        lines.append(f"{prediction.class_id} {cx:.6f} {cy:.6f} {box_width:.6f} {box_height:.6f} {confidence:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def draw_box(image: np.ndarray, detection: Detection, label: str, color: tuple[int, int, int], thickness: int = 2) -> None:
    """Draw one labelled box, keeping label text within the image canvas."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in detection.box)
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    text_thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, text_thickness)
    text_y = max(text_height + baseline + 2, y1)
    cv2.rectangle(image, (x1, text_y - text_height - baseline - 3), (x1 + text_width + 4, text_y + 2), color, -1)
    cv2.putText(image, label, (x1 + 2, text_y - baseline), font, scale, (0, 0, 0), text_thickness, cv2.LINE_AA)


def annotate_ground_truth(image: np.ndarray, gt: list[Detection], class_names: list[str], fn_indices: set[int] | None = None) -> np.ndarray:
    """Render green GT boxes, using yellow for objects missed by a model."""
    canvas = image.copy()
    for index, detection in enumerate(gt):
        missed = fn_indices is not None and index in fn_indices
        draw_box(canvas, detection, f"GT {safe_name(class_names, detection.class_id)}", FN_COLOR if missed else GT_COLOR)
    return canvas


def annotate_predictions(image: np.ndarray, predictions: list[Detection], gt: list[Detection], tp_matches: list[Match], confusion_matches: list[Match], class_names: list[str]) -> np.ndarray:
    """Render predictions plus yellow GT boxes that remain FN after matching."""
    canvas = image.copy()
    tp_indices = {match.pred_index for match in tp_matches}
    confusion_indices = {match.pred_index for match in confusion_matches}
    matched_gt = {match.gt_index for match in tp_matches}
    for index, detection in enumerate(gt):
        if index not in matched_gt:
            draw_box(canvas, detection, f"FN GT {safe_name(class_names, detection.class_id)}", FN_COLOR)
    for index, prediction in enumerate(predictions):
        status, color = ("TP", TP_COLOR) if index in tp_indices else ("CONF", CONFUSION_COLOR) if index in confusion_indices else ("FP", FP_COLOR)
        label = f"{status} {safe_name(class_names, prediction.class_id)} {prediction.confidence or 0.0:.2f}"
        draw_box(canvas, prediction, label, color)
    return canvas


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    """Add a compact white title strip to an image panel."""
    header_height = 34
    header = np.full((header_height, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([header, image])


def resize_panel(image: np.ndarray, height: int) -> np.ndarray:
    """Resize an image panel to a common height while retaining its aspect ratio."""
    scale = height / image.shape[0]
    return cv2.resize(image, (max(1, int(round(image.shape[1] * scale))), height), interpolation=cv2.INTER_AREA)


def save_side_by_side(path: Path, image: np.ndarray, gt: list[Detection], model_a: dict[str, Any], model_b: dict[str, Any], class_names: list[str], name_a: str, name_b: str) -> None:
    """Save original/GT, model-A, and model-B predictions in one inspection image."""
    panel_gt = annotate_ground_truth(image, gt, class_names)
    panel_a = annotate_predictions(image, model_a["predictions"], gt, model_a["tp_matches"], model_a["confusion_matches"], class_names)
    panel_b = annotate_predictions(image, model_b["predictions"], gt, model_b["tp_matches"], model_b["confusion_matches"], class_names)
    panels = [
        add_header(panel_gt, "Ground truth"),
        add_header(panel_a, f"{name_a}: TP={model_a['tp']} FP={model_a['fp']} FN={model_a['fn']}"),
        add_header(panel_b, f"{name_b}: TP={model_b['tp']} FP={model_b['fp']} FN={model_b['fn']}"),
    ]
    target_height = min(800, max(panel.shape[0] for panel in panels))
    combined = cv2.hconcat([resize_panel(panel, target_height) for panel in panels])
    if not cv2.imwrite(str(path), combined):
        raise OSError(f"Could not write comparison image: {path}")


def metric_row(counts: dict[str, int]) -> dict[str, float | int]:
    """Calculate TP/FP/FN-derived precision, recall, and F1 without divide-by-zero."""
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def initialise_counts(class_names: list[str]) -> dict[str, dict[str, int]]:
    """Create zeroed per-class GT/prediction/match counters."""
    return {name: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0} for name in class_names}


def analyse_image(gt: list[Detection], predictions: list[Detection], class_names: list[str], match_iou: float) -> dict[str, Any]:
    """Match one image and return totals, per-class counts, and class confusions."""
    tp_matches, confusion_matches, used_gt, used_pred = match_detections(gt, predictions, match_iou)
    per_class = initialise_counts(class_names)
    for detection in gt:
        per_class[safe_name(class_names, detection.class_id)]["gt"] += 1
    for detection in predictions:
        if 0 <= detection.class_id < len(class_names):
            per_class[safe_name(class_names, detection.class_id)]["pred"] += 1
    for match in tp_matches:
        per_class[safe_name(class_names, gt[match.gt_index].class_id)]["tp"] += 1
    for gt_index, detection in enumerate(gt):
        if gt_index not in used_gt:
            per_class[safe_name(class_names, detection.class_id)]["fn"] += 1
    for pred_index, detection in enumerate(predictions):
        if pred_index not in used_pred and 0 <= detection.class_id < len(class_names):
            per_class[safe_name(class_names, detection.class_id)]["fp"] += 1
    # A wrong-class overlap consumes both objects and is intentionally FP+FN.
    for match in confusion_matches:
        per_class[safe_name(class_names, gt[match.gt_index].class_id)]["fn"] += 1
        per_class[safe_name(class_names, predictions[match.pred_index].class_id)]["fp"] += 1
    confusions = [
        {
            "gt_class": safe_name(class_names, gt[match.gt_index].class_id),
            "pred_class": safe_name(class_names, predictions[match.pred_index].class_id),
            "iou": match.iou,
            "confidence": predictions[match.pred_index].confidence,
        }
        for match in confusion_matches
    ]
    total_tp = len(tp_matches)
    total_fp = len(predictions) - total_tp
    total_fn = len(gt) - total_tp
    return {
        "predictions": predictions,
        "tp_matches": tp_matches,
        "confusion_matches": confusion_matches,
        "per_class": per_class,
        "confusions": confusions,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "used_gt": used_gt,
        "used_pred": used_pred,
    }


def detection_csv_fields(prefix: str, detection: Detection | None, class_names: list[str]) -> dict[str, Any]:
    """Return CSV-safe class and xyxy fields for one optional detection."""
    if detection is None:
        return {
            f"{prefix}_class": "",
            f"{prefix}_x1": "",
            f"{prefix}_y1": "",
            f"{prefix}_x2": "",
            f"{prefix}_y2": "",
        }
    x1, y1, x2, y2 = detection.box
    return {
        f"{prefix}_class": safe_name(class_names, detection.class_id),
        f"{prefix}_x1": round(x1, 3),
        f"{prefix}_y1": round(y1, 3),
        f"{prefix}_x2": round(x2, 3),
        f"{prefix}_y2": round(y2, 3),
    }


def is_threshold_sensitive(confidence: float | None, threshold: float) -> bool:
    """Identify predictions within a small tolerance of the selected threshold."""
    return confidence is not None and abs(confidence - threshold) <= max(0.02, threshold * 0.1)


def build_error_record(
    model_name: str,
    image_name: str,
    error_type: str,
    gt_detection: Detection | None,
    prediction: Detection | None,
    match_iou: float | None,
    reason_hint: str,
    class_names: list[str],
) -> dict[str, Any]:
    """Create one TP, FP, or FN record with both optional boxes represented."""
    record = {
        "model": model_name,
        "image_name": image_name,
        "error_type": error_type,
        "confidence": "" if prediction is None or prediction.confidence is None else round(prediction.confidence, 6),
        "match_iou": "" if match_iou is None else round(match_iou, 6),
        "reason_hint": reason_hint,
    }
    record.update(detection_csv_fields("gt", gt_detection, class_names))
    record.update(detection_csv_fields("pred", prediction, class_names))
    return record


def build_error_records(
    model_name: str,
    image_name: str,
    gt: list[Detection],
    result: dict[str, Any],
    class_names: list[str],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    """Build auditable TP/FP/FN records while preserving class-confusion links.

    A class-confusion pair remains two accounting records: one FP for the
    predicted class and one FN for the ground-truth class. This mirrors the
    existing aggregate counts instead of changing the matching definition.
    Reason hints are triage assistance only, not a final defect diagnosis.
    """
    predictions: list[Detection] = result["predictions"]
    records: list[dict[str, Any]] = []
    tp_gt = {match.gt_index for match in result["tp_matches"]}
    tp_pred = {match.pred_index for match in result["tp_matches"]}
    confusion_gt = {match.gt_index for match in result["confusion_matches"]}
    confusion_pred = {match.pred_index for match in result["confusion_matches"]}

    for match in result["tp_matches"]:
        records.append(build_error_record(model_name, image_name, "TP", gt[match.gt_index], predictions[match.pred_index], match.iou, "", class_names))
    for match in result["confusion_matches"]:
        gt_detection, prediction = gt[match.gt_index], predictions[match.pred_index]
        records.append(build_error_record(model_name, image_name, "FP", gt_detection, prediction, match.iou, "class_confusion", class_names))
        records.append(build_error_record(model_name, image_name, "FN", gt_detection, prediction, match.iou, "class_confusion", class_names))
    for pred_index, prediction in enumerate(predictions):
        if pred_index in tp_pred or pred_index in confusion_pred:
            continue
        reason = "threshold_sensitive" if is_threshold_sensitive(prediction.confidence, confidence_threshold) else "background_or_unlabeled_object"
        records.append(build_error_record(model_name, image_name, "FP", None, prediction, None, reason, class_names))
    for gt_index, gt_detection in enumerate(gt):
        if gt_index in tp_gt or gt_index in confusion_gt:
            continue
        records.append(build_error_record(model_name, image_name, "FN", gt_detection, None, None, "missed_detection", class_names))
    return records


def save_error_details_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Save complete TP/FP/FN box-level matching evidence for both models."""
    write_csv(
        path,
        ["model", "image_name", "error_type", "gt_class", "pred_class", "confidence", "gt_x1", "gt_y1", "gt_x2", "gt_y2", "pred_x1", "pred_y1", "pred_x2", "pred_y2", "match_iou"],
        records,
    )


def save_open_circuit_errors_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Save FP/FN records relevant to open_circuit with a diagnostic hint."""
    selected = [
        record for record in records
        if (record["error_type"] == "FP" and record["pred_class"] == "open_circuit")
        or (record["error_type"] == "FN" and record["gt_class"] == "open_circuit")
    ]
    write_csv(path, ["image_name", "error_type", "gt_class", "pred_class", "confidence", "match_iou", "reason_hint"], selected)


def build_image_error_summary(model_name: str, image_name: str, result: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    """Summarise one model/image pair for review-priority sorting."""
    row: dict[str, Any] = {
        "model": model_name,
        "image_name": image_name,
        "tp": result["tp"],
        "fp": result["fp"],
        "fn": result["fn"],
        "class_confusion_count": len(result["confusion_matches"]),
    }
    for class_name in class_names:
        row[f"{class_name}_fp"] = result["per_class"][class_name]["fp"]
        row[f"{class_name}_fn"] = result["per_class"][class_name]["fn"]
    return row


def save_image_error_summary(path: Path, rows: list[dict[str, Any]], class_names: list[str]) -> list[dict[str, Any]]:
    """Sort and save image-level error counts, placing costly reviews first."""
    ordered = sorted(rows, key=lambda row: (-row["fn"], -row["fp"], -row["class_confusion_count"], row["image_name"], row["model"]))
    fields = ["model", "image_name", "tp", "fp", "fn"]
    for class_name in class_names:
        fields.extend((f"{class_name}_fp", f"{class_name}_fn"))
    fields.append("class_confusion_count")
    write_csv(path, fields, ordered)
    return ordered


def render_error_analysis_images(
    error_root: Path,
    model_name: str,
    image_name: str,
    image: np.ndarray,
    records: list[dict[str, Any]],
    class_names: list[str],
) -> None:
    """Render error-only inspection images into review folders without copying originals."""
    errors = [record for record in records if record["error_type"] in {"FP", "FN"}]
    if not errors:
        return
    model_root = error_root / model_name
    categories = {"all_errors"}
    if any(record["error_type"] == "FP" and record["pred_class"] == "open_circuit" for record in errors):
        categories.add("open_circuit_fp")
    if any(record["error_type"] == "FN" and record["gt_class"] == "open_circuit" for record in errors):
        categories.add("open_circuit_fn")
    if any(record["reason_hint"] == "class_confusion" for record in errors):
        categories.add("class_confusion")

    canvas = image.copy()
    drawn: set[tuple[str, str, float, float, float, float]] = set()
    for record in errors:
        is_confusion = record["reason_hint"] == "class_confusion"
        if record["error_type"] == "FN" and record["gt_class"]:
            key = ("gt", record["gt_class"], record["gt_x1"], record["gt_y1"], record["gt_x2"], record["gt_y2"])
            if key not in drawn:
                drawn.add(key)
                class_id = class_names.index(record["gt_class"]) if record["gt_class"] in class_names else 0
                box = Detection(class_id, (float(record["gt_x1"]), float(record["gt_y1"]), float(record["gt_x2"]), float(record["gt_y2"])))
                draw_box(canvas, box, f"GT: {record['gt_class']}", CONFUSION_COLOR if is_confusion else FN_COLOR, 3)
        if record["error_type"] == "FP" and record["pred_class"]:
            key = ("pred", record["pred_class"], record["pred_x1"], record["pred_y1"], record["pred_x2"], record["pred_y2"])
            if key not in drawn:
                drawn.add(key)
                confidence = float(record["confidence"]) if record["confidence"] != "" else None
                class_id = class_names.index(record["pred_class"]) if record["pred_class"] in class_names else 0
                box = Detection(class_id, (float(record["pred_x1"]), float(record["pred_y1"]), float(record["pred_x2"]), float(record["pred_y2"])), confidence)
                label = f"Pred: {record['pred_class']} {confidence:.2f}" if confidence is not None else f"Pred: {record['pred_class']}"
                draw_box(canvas, box, label, CONFUSION_COLOR if is_confusion else FP_COLOR, 3)
    fp_count = sum(record["error_type"] == "FP" for record in errors)
    fn_count = sum(record["error_type"] == "FN" for record in errors)
    confusion_count = sum(record["reason_hint"] == "class_confusion" and record["error_type"] == "FP" for record in errors)
    rendered = add_header(canvas, f"{image_name} | FP={fp_count} FN={fn_count} Class confusion={confusion_count}")
    for category in categories:
        target = model_root / category
        target.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target / image_name), rendered):
            raise OSError(f"Could not write error analysis image: {target / image_name}")


def print_error_analysis_summary(ordered_rows: list[dict[str, Any]], class_names: list[str], model_names: tuple[str, str]) -> None:
    """Print concise per-model error totals and the ten highest-priority images."""
    print("\nError analysis summary")
    for model_name in model_names:
        rows = [row for row in ordered_rows if row["model"] == model_name]
        open_fp = sum(row.get("open_circuit_fp", 0) for row in rows)
        open_fn = sum(row.get("open_circuit_fn", 0) for row in rows)
        confusion = sum(row["class_confusion_count"] for row in rows)
        review_count = sum(1 for row in rows if row["fp"] or row["fn"])
        print(f"\nModel: {model_name}")
        print(f"Open circuit FP: {open_fp}")
        print(f"Open circuit FN: {open_fn}")
        print(f"Class confusion: {confusion}")
        print(f"Images requiring review: {review_count}")
        top_rows = [row for row in rows if row["fp"] or row["fn"]][:10]
        print("Top review images")
        print_table(
            ["Image name", "FP", "FN", "Confusion"],
            [[row["image_name"], str(row["fp"]), str(row["fn"]), str(row["class_confusion_count"])] for row in top_rows],
        )


def merge_counts(target: dict[str, dict[str, int]], source: dict[str, dict[str, int]]) -> None:
    """Add one image's per-class integer counts to model-wide counts."""
    for class_name, counts in source.items():
        for key, value in counts.items():
            target[class_name][key] += value


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    """Write UTF-8 CSV using the standard library, without requiring pandas."""
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: float | int) -> str:
    """Format counts and rates for terminal tables."""
    return str(value) if isinstance(value, int) else f"{value:.4f}"


def better(value_a: float | int, value_b: float | int, lower_is_better: bool = False) -> str:
    """Choose an unambiguous better model label, retaining ties."""
    if value_a == value_b:
        return "Tie"
    a_wins = value_a < value_b if lower_is_better else value_a > value_b
    return "A" if a_wins else "B"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a dependency-free aligned terminal table."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def run_model(model_name: str, model_path: Path, images: list[Path], args: argparse.Namespace) -> dict[str, list[Detection]]:
    """Run one model once and return predictions keyed by image filename stem."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is not installed. Run the project's requirements installation first.") from error
    image_parents = {image.parent.resolve() for image in images}
    if len(image_parents) != 1:
        raise RuntimeError("All requested images must be in one directory for filename-preserving YOLO inference.")
    # A list source makes some Ultralytics versions rename result.path to image0.jpg,
    # image1.jpg, etc. Directory input retains actual filenames for stem matching.
    image_directory = next(iter(image_parents))
    model = YOLO(str(model_path))
    raw_results = model.predict(
        source=str(image_directory),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
        stream=False,
        save=False,
    )
    results = list(raw_results)
    predictions_by_image: dict[str, list[Detection]] = {}
    total_raw_boxes = 0
    total_converted = 0

    print(f"[DEBUG] Model: {model_name}")
    print(f"[DEBUG] Result objects: {len(results)}")
    for index, result in enumerate(results):
        result_path = Path(str(getattr(result, "path", "")))
        image_key = result_path.stem
        if not image_key:
            raise RuntimeError(f"YOLO returned a result without an image path for model '{model_name}'.")
        raw_count = len(result.boxes) if getattr(result, "boxes", None) is not None else 0
        detections = result_to_detections(result)
        total_raw_boxes += raw_count
        total_converted += len(detections)
        if image_key in predictions_by_image:
            raise RuntimeError(f"Duplicate image key returned by YOLO for '{model_name}': {image_key}")
        predictions_by_image[image_key] = detections
        if index < 3:
            print(f"[DEBUG] {result_path.name}: raw={raw_count}, converted={len(detections)}")

    print(f"[DEBUG] Total raw boxes: {total_raw_boxes}")
    print(f"[DEBUG] Total converted predictions: {total_converted}")
    if len(results) != len(images):
        raise RuntimeError(f"YOLO returned {len(results)} results for {len(images)} requested images: {model_path}")
    if total_raw_boxes != total_converted:
        raise RuntimeError(f"Prediction conversion mismatch for '{model_name}': raw={total_raw_boxes}, converted={total_converted}")

    requested_keys = {path.stem for path in images}
    if len(requested_keys) != len(images):
        raise RuntimeError("Input images have duplicate filename stems; stem-based matching is ambiguous.")
    returned_keys = set(predictions_by_image)
    missing_keys = sorted(requested_keys - returned_keys)
    unexpected_keys = sorted(returned_keys - requested_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(f"Prediction/image key mismatch for '{model_name}': missing={missing_keys}, unexpected={unexpected_keys}")
    return predictions_by_image


def main() -> int:
    """Run prediction, manual evaluation, reporting, and visual inspection output."""
    args = parse_args()
    try:
        paths = validate_args(args)
        class_names = load_class_names(paths["data"])
        images = find_images(paths["images"])
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    output_dir = create_output_dir(paths["project"], args.run_name)
    model_dirs = {args.name_a: output_dir / args.name_a, args.name_b: output_dir / args.name_b}
    for directory in model_dirs.values():
        (directory / "images").mkdir(parents=True)
        (directory / "labels").mkdir(parents=True)
    side_by_side_dir = output_dir / "side_by_side"
    side_by_side_dir.mkdir()
    error_analysis_dir = output_dir / "error_analysis"
    for model_name in (args.name_a, args.name_b):
        for category in ("open_circuit_fp", "open_circuit_fn", "class_confusion", "all_errors"):
            (error_analysis_dir / model_name / category).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Images: {len(images)} | imgsz={args.imgsz}, conf={args.conf}, NMS iou={args.iou}, match iou={args.match_iou}")
    print(f"[INFO] Predicting {args.name_a}: {paths['model_a']}")
    try:
        predictions_a = run_model(args.name_a, paths["model_a"], images, args)
        print(f"[INFO] Predicting {args.name_b}: {paths['model_b']}")
        predictions_b = run_model(args.name_b, paths["model_b"], images, args)
    except Exception as error:  # Surface CUDA/Ultralytics errors with the selected model context.
        print(f"[ERROR] Prediction failed: {error}", file=sys.stderr)
        return 1

    totals = {args.name_a: initialise_counts(class_names), args.name_b: initialise_counts(class_names)}
    image_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    image_error_rows: list[dict[str, Any]] = []
    overall = {args.name_a: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}, args.name_b: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}}

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARNING] Could not read image; skipped: {image_path}")
            continue
        height, width = image.shape[:2]
        gt = read_yolo_labels(paths["labels"] / f"{image_path.stem}.txt", width, height, len(class_names))
        image_key = image_path.stem
        if image_key not in predictions_a:
            raise RuntimeError(f"Missing predictions for {args.name_a}: {image_path.name}")
        if image_key not in predictions_b:
            raise RuntimeError(f"Missing predictions for {args.name_b}: {image_path.name}")
        image_results = {
            args.name_a: analyse_image(gt, predictions_a[image_key], class_names, args.match_iou),
            args.name_b: analyse_image(gt, predictions_b[image_key], class_names, args.match_iou),
        }
        for model_name, result in image_results.items():
            destination = model_dirs[model_name]
            rendered = annotate_predictions(image, result["predictions"], gt, result["tp_matches"], result["confusion_matches"], class_names)
            if not cv2.imwrite(str(destination / "images" / image_path.name), rendered):
                raise OSError(f"Could not write annotated prediction image for {image_path.name}")
            save_prediction_labels(destination / "labels" / f"{image_path.stem}.txt", result["predictions"], width, height)
            merge_counts(totals[model_name], result["per_class"])
            for key in overall[model_name]:
                overall[model_name][key] += result[key] if key in result else len(gt) if key == "gt" else len(result["predictions"]) if key == "pred" else 0
            for class_name, counts in result["per_class"].items():
                row_metrics = metric_row(counts)
                image_rows.append({
                    "image": image_path.name, "model": model_name, "class": class_name,
                    "gt_count": row_metrics["gt"], "pred_count": row_metrics["pred"],
                    "tp": row_metrics["tp"], "fp": row_metrics["fp"], "fn": row_metrics["fn"],
                    "precision": row_metrics["precision"], "recall": row_metrics["recall"], "f1": row_metrics["f1"],
                    "confusions": "; ".join(f"{item['gt_class']} -> {item['pred_class']}" for item in result["confusions"] if item["gt_class"] == class_name or item["pred_class"] == class_name),
                    "notes": "",
                })
            for item in result["confusions"]:
                confusion_rows.append({"image": image_path.name, "model": model_name, **item})
            image_records = build_error_records(model_name, image_path.name, gt, result, class_names, args.conf)
            error_records.extend(image_records)
            image_error_rows.append(build_image_error_summary(model_name, image_path.name, result, class_names))
            render_error_analysis_images(error_analysis_dir, model_name, image_path.name, image, image_records, class_names)
        save_side_by_side(side_by_side_dir / image_path.name, image, gt, image_results[args.name_a], image_results[args.name_b], class_names, args.name_a, args.name_b)

    # Overall GT/pred totals are direct counts; TP/FP/FN came from matching.
    for model_name in (args.name_a, args.name_b):
        overall[model_name]["gt"] = sum(row["gt"] for row in totals[model_name].values())
        overall[model_name]["pred"] = sum(row["pred"] for row in totals[model_name].values())

    class_rows: list[dict[str, Any]] = []
    for model_name in (args.name_a, args.name_b):
        for class_name in class_names:
            class_rows.append({"model": model_name, "class": class_name, **metric_row(totals[model_name][class_name])})
    write_csv(output_dir / "image_details.csv", ["image", "model", "class", "gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1", "confusions", "notes"], image_rows)
    write_csv(output_dir / "class_summary.csv", ["model", "class", "gt", "pred", "tp", "fp", "fn", "precision", "recall", "f1"], class_rows)
    write_csv(output_dir / "confusion_details.csv", ["image", "model", "gt_class", "pred_class", "iou", "confidence"], confusion_rows)
    save_error_details_csv(output_dir / "error_details.csv", error_records)
    save_open_circuit_errors_csv(output_dir / "open_circuit_errors.csv", error_records)
    ordered_error_rows = save_image_error_summary(output_dir / "image_error_summary.csv", image_error_rows, class_names)

    overall_metrics = {name: metric_row(counts) for name, counts in overall.items()}
    print("\nOverall comparison")
    print_table(
        ["Model", "GT", "Pred", "TP", "FP", "FN", "Precision", "Recall", "F1"],
        [[name, *(format_value(overall_metrics[name][key]) for key in ("gt", "pred", "tp", "fp", "fn", "precision", "recall", "f1"))] for name in (args.name_a, args.name_b)],
    )
    print("\nClass comparison")
    comparison_rows: list[list[str]] = []
    for class_name in class_names:
        a_metrics = metric_row(totals[args.name_a][class_name])
        b_metrics = metric_row(totals[args.name_b][class_name])
        for metric in ("tp", "fp", "fn", "precision", "recall"):
            comparison_rows.append([class_name, metric.upper() if metric in {"tp", "fp", "fn"} else metric.title(), format_value(a_metrics[metric]), format_value(b_metrics[metric]), better(a_metrics[metric], b_metrics[metric], metric in {"fp", "fn"}).replace("A", args.name_a).replace("B", args.name_b)])
    print_table(["Class", "Metric", args.name_a, args.name_b, "Better"], comparison_rows)
    print("\nMissed defects comparison")
    missed_rows = []
    for class_name in class_names:
        a_fn, b_fn = totals[args.name_a][class_name]["fn"], totals[args.name_b][class_name]["fn"]
        missed_rows.append([class_name, str(a_fn), str(b_fn), better(a_fn, b_fn, True).replace("A", args.name_a).replace("B", args.name_b)])
    missed_rows.append(["total", str(overall[args.name_a]["fn"]), str(overall[args.name_b]["fn"]), better(overall[args.name_a]["fn"], overall[args.name_b]["fn"], True).replace("A", args.name_a).replace("B", args.name_b)])
    print_table(["Class", f"{args.name_a} FN", f"{args.name_b} FN", "Better"], missed_rows)
    print_error_analysis_summary(ordered_error_rows, class_names, (args.name_a, args.name_b))
    print(f"\n[INFO] CSV and images saved under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
