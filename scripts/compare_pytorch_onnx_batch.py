from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
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

from model.defect_info import Detection
from scripts.compare_pytorch_onnx import (
    DetectionMatch,
    draw_detections,
    load_class_names,
    match_detections,
    pytorch_result_to_detections,
    run_pytorch_once,
    synchronize_cuda,
)
from service.onnx_detector import OnnxDetector, detection_to_dict


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PASS_CONFIDENCE_DIFF_MAX = 0.01
PASS_BBOX_IOU_MIN = 0.99
CSV_ENCODING = "utf-8-sig"


@dataclass(frozen=True, slots=True)
class TimingSample:
    backend: str
    repeat_index: int
    preprocess_ms: float | None
    inference_ms: float | None
    postprocess_ms: float | None
    total_ms: float
    provider: str


@dataclass(frozen=True, slots=True)
class TimedDetectionsResult:
    detections: list[Detection]
    timings: list[TimingSample]
    providers: list[str] = field(default_factory=list)
    input_name: str | None = None
    output_name: str | None = None
    input_shape: list[Any] | None = None
    output_shape: list[int] | None = None


@dataclass(frozen=True, slots=True)
class ImageComparison:
    image_name: str
    image_path: str
    status: str
    status_reason: str
    pytorch_detection_count: int | None = None
    onnx_detection_count: int | None = None
    matched_count: int | None = None
    pytorch_only_count: int | None = None
    onnx_only_count: int | None = None
    classes_match: bool | None = None
    confidence_diff_mean: float | None = None
    confidence_diff_max: float | None = None
    bbox_iou_mean: float | None = None
    bbox_iou_min: float | None = None
    pytorch_timing: dict[str, dict[str, float | None]] = field(default_factory=dict)
    onnx_timing: dict[str, dict[str, float | None]] = field(default_factory=dict)
    speedup: float | None = None
    onnx_provider: str = ""
    error_message: str = ""
    pytorch_detections: list[dict[str, Any]] = field(default_factory=list)
    onnx_detections: list[dict[str, Any]] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timing_rows: list[TimingSample] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BatchRunResult:
    summary: dict[str, Any]
    image_results: list[ImageComparison]
    timing_rows: list[TimingSample]
    exit_code: int


PytorchRunner = Callable[[Path], TimedDetectionsResult]
OnnxRunner = Callable[[Path], TimedDetectionsResult]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch YOLO and ONNX Runtime outputs over an image batch.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--pytorch-model", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--save-all-images", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--extensions", default=",".join(SUPPORTED_EXTENSIONS))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args(argv)


def collect_images(
    image_dir: Path,
    extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS,
    recursive: bool = False,
    max_images: int | None = None,
    seed: int | None = None,
) -> list[Path]:
    normalized = tuple(_normalize_extension(extension) for extension in extensions)
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in normalized)
    if seed is not None:
        rng = random.Random(seed)
        images = images[:]
        rng.shuffle(images)
    if max_images is not None:
        images = images[:max_images]
    return images


def validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not args.images.exists() or not args.images.is_dir():
        errors.append(f"Image directory does not exist: {args.images}")
    if not args.pytorch_model.exists() or not args.pytorch_model.is_file():
        errors.append(f"PyTorch model file does not exist: {args.pytorch_model}")
    if not args.onnx_model.exists() or not args.onnx_model.is_file():
        errors.append(f"ONNX model file does not exist: {args.onnx_model}")
    if args.repeat < 1:
        errors.append("--repeat must be >= 1")
    if args.warmup < 0:
        errors.append("--warmup must be >= 0")
    if args.imgsz <= 0:
        errors.append("--imgsz must be > 0")
    if not 0.0 <= args.conf <= 1.0:
        errors.append("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        errors.append("--iou must be between 0 and 1")
    if not 0.0 <= args.match_iou <= 1.0:
        errors.append("--match-iou must be between 0 and 1")
    if args.max_images is not None and args.max_images < 1:
        errors.append("--max-images must be >= 1")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        errors.append(f"Cannot create output directory {args.output}: {exc}")
    return errors


def extensions_from_arg(value: str) -> tuple[str, ...]:
    extensions = tuple(_normalize_extension(part.strip()) for part in value.split(",") if part.strip())
    return extensions or SUPPORTED_EXTENSIONS


def timing_stats(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None, "std_ms": None, "p95_ms": None}
    values = sorted(float(value) for value in samples)
    return {
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "min_ms": float(values[0]),
        "max_ms": float(values[-1]),
        "std_ms": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "p95_ms": percentile(values, 95.0),
    }


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile_value / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[int(rank)])
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return float(lower_value + (upper_value - lower_value) * (rank - lower))


def summarize_backend_timings(timings: list[TimingSample]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for field_name in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"):
        values = [float(value) for sample in timings if (value := getattr(sample, field_name)) is not None]
        result[field_name] = timing_stats(values)
    return result


def compare_detection_sets(
    image_path: Path,
    pt_result: TimedDetectionsResult,
    onnx_result: TimedDetectionsResult,
    match_iou: float,
) -> ImageComparison:
    matches = match_detections(pt_result.detections, onnx_result.detections, match_iou=match_iou)
    matched = [match for match in matches if match.status == "MATCHED" and match.pt is not None and match.onnx is not None]
    pt_only = [match for match in matches if match.status == "PT_ONLY"]
    onnx_only = [match for match in matches if match.status == "ONNX_ONLY"]
    conf_diffs = [abs(match.pt.confidence - match.onnx.confidence) for match in matched]
    bbox_ious = [match.iou for match in matched]
    classes_match = all(match.pt.class_id == match.onnx.class_id for match in matched)

    confidence_diff_mean = float(statistics.fmean(conf_diffs)) if conf_diffs else 0.0
    confidence_diff_max = float(max(conf_diffs)) if conf_diffs else 0.0
    bbox_iou_mean = float(statistics.fmean(bbox_ious)) if bbox_ious else 0.0
    bbox_iou_min = float(min(bbox_ious)) if bbox_ious else 1.0
    pt_timing = summarize_backend_timings(pt_result.timings)
    onnx_timing = summarize_backend_timings(onnx_result.timings)
    speedup = calculate_speedup(_stat(pt_timing, "total_ms", "mean_ms"), _stat(onnx_timing, "total_ms", "mean_ms"))

    status, reason, warnings = judge_status(
        len(pt_result.detections),
        len(onnx_result.detections),
        len(matched),
        len(pt_only),
        len(onnx_only),
        classes_match,
        confidence_diff_max,
        bbox_iou_min,
    )

    return ImageComparison(
        image_name=image_path.name,
        image_path=str(image_path),
        status=status,
        status_reason=reason,
        pytorch_detection_count=len(pt_result.detections),
        onnx_detection_count=len(onnx_result.detections),
        matched_count=len(matched),
        pytorch_only_count=len(pt_only),
        onnx_only_count=len(onnx_only),
        classes_match=classes_match,
        confidence_diff_mean=confidence_diff_mean,
        confidence_diff_max=confidence_diff_max,
        bbox_iou_mean=bbox_iou_mean,
        bbox_iou_min=bbox_iou_min,
        pytorch_timing=pt_timing,
        onnx_timing=onnx_timing,
        speedup=speedup,
        onnx_provider=onnx_result.providers[0] if onnx_result.providers else "",
        pytorch_detections=[detection_to_dict(detection) for detection in pt_result.detections],
        onnx_detections=[detection_to_dict(detection) for detection in onnx_result.detections],
        matches=[match_to_dict(match) for match in matches],
        warnings=warnings,
        timing_rows=[*pt_result.timings, *onnx_result.timings],
    )


def judge_status(
    pt_count: int,
    onnx_count: int,
    matched_count: int,
    pt_only_count: int,
    onnx_only_count: int,
    classes_match: bool,
    confidence_diff_max: float,
    bbox_iou_min: float,
) -> tuple[str, str, list[str]]:
    if pt_count == 0 and onnx_count == 0:
        return "PASS", "Both backends returned zero detections.", []
    if pt_only_count > 0 or onnx_only_count > 0 or not classes_match:
        reasons = []
        if pt_only_count > 0:
            reasons.append(f"pytorch_only_count={pt_only_count}")
        if onnx_only_count > 0:
            reasons.append(f"onnx_only_count={onnx_only_count}")
        if not classes_match:
            reasons.append("classes_match=false")
        return "FAIL", "; ".join(reasons), reasons

    warnings = []
    if confidence_diff_max > PASS_CONFIDENCE_DIFF_MAX:
        warnings.append(f"confidence_diff_max={confidence_diff_max:.6f} > {PASS_CONFIDENCE_DIFF_MAX}")
    if matched_count > 0 and bbox_iou_min < PASS_BBOX_IOU_MIN:
        warnings.append(f"bbox_iou_min={bbox_iou_min:.6f} < {PASS_BBOX_IOU_MIN}")
    if warnings:
        return "WARNING", "; ".join(warnings), warnings
    return "PASS", "Detection counts, classes, confidence, and boxes are within thresholds.", []


def calculate_speedup(pytorch_ms: float | None, onnx_ms: float | None) -> float | None:
    if pytorch_ms is None or onnx_ms is None or onnx_ms <= 0:
        return None
    return float(pytorch_ms / onnx_ms)


def final_status(results: list[ImageComparison]) -> str:
    statuses = {result.status for result in results}
    if "ERROR" in statuses:
        return "ERROR"
    if "FAIL" in statuses:
        return "FAIL"
    if "WARNING" in statuses:
        return "WARNING"
    return "PASS"


def build_summary(
    args: argparse.Namespace,
    images: list[Path],
    results: list[ImageComparison],
    timing_rows: list[TimingSample],
    environment: dict[str, Any],
) -> dict[str, Any]:
    processed = [result for result in results if result.status != "ERROR"]
    matched_conf_diffs = [
        value
        for result in processed
        for match in result.matches
        if (value := match.get("confidence_diff_abs")) is not None
    ]
    matched_ious = [
        value
        for result in processed
        for match in result.matches
        if match.get("status") == "MATCHED" and (value := match.get("bbox_iou")) is not None
    ]
    pt_total = [sample.total_ms for sample in timing_rows if sample.backend == "pytorch"]
    onnx_total = [sample.total_ms for sample in timing_rows if sample.backend == "onnx"]
    speedups = [result.speedup for result in processed if result.speedup is not None]

    status = final_status(results)
    return {
        "run_info": {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "output": str(args.output),
            "final_status": status,
        },
        "models": {
            "pytorch": str(args.pytorch_model),
            "onnx": str(args.onnx_model),
        },
        "dataset": {
            "image_directory": str(args.images),
            "total_discovered_images": len(images),
            "extensions": list(extensions_from_arg(args.extensions)),
            "recursive": bool(args.recursive),
        },
        "onnx_runtime": {
            "available_providers": environment.get("onnxruntime_available_providers", []),
            "selected_providers": sorted({provider for sample in timing_rows if (provider := sample.provider)}),
        },
        "thresholds": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "match_iou": args.match_iou,
            "confidence_diff_max": PASS_CONFIDENCE_DIFF_MAX,
            "bbox_iou_min": PASS_BBOX_IOU_MIN,
        },
        "counts": {
            "total_images": len(images),
            "processed_images": len(processed),
            "pass_count": _count_status(results, "PASS"),
            "warning_count": _count_status(results, "WARNING"),
            "fail_count": _count_status(results, "FAIL"),
            "error_count": _count_status(results, "ERROR"),
            "pass_rate": _rate(_count_status(results, "PASS"), len(results)),
            "warning_rate": _rate(_count_status(results, "WARNING"), len(results)),
            "fail_rate": _rate(_count_status(results, "FAIL"), len(results)),
        },
        "detection_summary": {
            "total_pytorch_detections": sum(result.pytorch_detection_count or 0 for result in processed),
            "total_onnx_detections": sum(result.onnx_detection_count or 0 for result in processed),
            "total_matched_detections": sum(result.matched_count or 0 for result in processed),
            "total_pytorch_only": sum(result.pytorch_only_count or 0 for result in processed),
            "total_onnx_only": sum(result.onnx_only_count or 0 for result in processed),
        },
        "accuracy_difference_summary": {
            "overall_confidence_diff_mean": _mean_or_none(matched_conf_diffs),
            "overall_confidence_diff_max": max(matched_conf_diffs) if matched_conf_diffs else None,
            "overall_bbox_iou_mean": _mean_or_none(matched_ious),
            "overall_bbox_iou_min": min(matched_ious) if matched_ious else None,
        },
        "timing_summary": {
            "pytorch_total_mean_ms": timing_stats(pt_total)["mean_ms"],
            "pytorch_total_median_ms": timing_stats(pt_total)["median_ms"],
            "pytorch_total_p95_ms": timing_stats(pt_total)["p95_ms"],
            "onnx_total_mean_ms": timing_stats(onnx_total)["mean_ms"],
            "onnx_total_median_ms": timing_stats(onnx_total)["median_ms"],
            "onnx_total_p95_ms": timing_stats(onnx_total)["p95_ms"],
            "speedup_mean": _mean_or_none(speedups),
            "speedup_median": statistics.median(speedups) if speedups else None,
        },
        "mismatch_images": [result.image_name for result in results if result.status in {"WARNING", "FAIL"}],
        "warning_images": [result.image_name for result in results if result.status == "WARNING"],
        "fail_images": [result.image_name for result in results if result.status == "FAIL"],
        "error_images": [result.image_name for result in results if result.status == "ERROR"],
        "final_status": status,
    }


def run_batch(
    args: argparse.Namespace,
    pytorch_runner: PytorchRunner | None = None,
    onnx_runner: OnnxRunner | None = None,
    class_names: dict[int, str] | None = None,
) -> BatchRunResult:
    validation_errors = validate_args(args)
    extensions = extensions_from_arg(args.extensions)
    images = collect_images(args.images, extensions, args.recursive, args.max_images, args.seed) if not validation_errors else []
    if not images and not validation_errors:
        validation_errors.append(f"No supported images found in {args.images}")
    if validation_errors:
        for error in validation_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return BatchRunResult({}, [], [], 2)

    class_names = class_names or load_class_names()
    args.output.mkdir(parents=True, exist_ok=True)
    _create_output_dirs(args.output)
    environment = collect_environment(args)

    if pytorch_runner is None or onnx_runner is None:
        try:
            pytorch_runner, onnx_runner = build_real_runners(args, images[0], class_names)
        except Exception as exc:
            print(f"ERROR: failed to initialize models: {exc}", file=sys.stderr)
            return BatchRunResult({}, [], [], 2)

    print_startup(args, images, environment)

    image_results: list[ImageComparison] = []
    timing_rows: list[TimingSample] = []
    for index, image_path in enumerate(images, start=1):
        try:
            pt_result = pytorch_runner(image_path)
            onnx_result = onnx_runner(image_path)
            comparison = compare_detection_sets(image_path, pt_result, onnx_result, args.match_iou)
            timing_rows.extend(pt_result.timings)
            timing_rows.extend(onnx_result.timings)
            write_per_image_json(args.output, comparison)
            maybe_write_side_by_side(args.output, image_path, comparison, class_names, args.save_all_images)
            image_results.append(comparison)
            print_progress(index, len(images), comparison)
        except Exception as exc:
            comparison = ImageComparison(
                image_name=image_path.name,
                image_path=str(image_path),
                status="ERROR",
                status_reason="Image-level exception occurred.",
                error_message=f"{type(exc).__name__}: {exc}",
            )
            image_results.append(comparison)
            write_per_image_json(args.output, comparison)
            print_progress(index, len(images), comparison)

    summary = build_summary(args, images, image_results, timing_rows, environment)
    write_reports(args.output, summary, image_results, timing_rows, environment, args)
    print_final_summary(summary, args.output)

    exit_code = exit_code_for_status(summary["final_status"], args.fail_on_warning)
    return BatchRunResult(summary, image_results, timing_rows, exit_code)


def build_real_runners(
    args: argparse.Namespace,
    warmup_image: Path,
    class_names: dict[int, str],
) -> tuple[PytorchRunner, OnnxRunner]:
    from ultralytics import YOLO

    pytorch_model = YOLO(str(args.pytorch_model))
    onnx_detector = OnnxDetector(args.onnx_model, imgsz=args.imgsz, conf=args.conf, iou=args.iou, class_names=class_names)

    for _ in range(args.warmup):
        run_pytorch_once(pytorch_model, warmup_image, args.imgsz, args.conf, args.iou, str(args.device))
    for _ in range(args.warmup):
        onnx_detector.detect_timed(warmup_image)

    def run_pt(image_path: Path) -> TimedDetectionsResult:
        detections: list[Detection] = []
        timings: list[TimingSample] = []
        for repeat_index in range(args.repeat):
            detections, timing = run_pytorch_once(pytorch_model, image_path, args.imgsz, args.conf, args.iou, str(args.device))
            timings.append(
                TimingSample(
                    backend="pytorch",
                    repeat_index=repeat_index,
                    preprocess_ms=timing.get("preprocess_ms"),
                    inference_ms=timing.get("inference_ms"),
                    postprocess_ms=timing.get("postprocess_ms"),
                    total_ms=float(timing["total_ms"]),
                    provider=str(args.device),
                )
            )
        return TimedDetectionsResult(detections=detections, timings=timings)

    def run_onnx(image_path: Path) -> TimedDetectionsResult:
        timed = None
        timings: list[TimingSample] = []
        for repeat_index in range(args.repeat):
            timed = onnx_detector.detect_timed(image_path)
            timings.append(
                TimingSample(
                    backend="onnx",
                    repeat_index=repeat_index,
                    preprocess_ms=timed.preprocess_ms,
                    inference_ms=timed.inference_ms,
                    postprocess_ms=timed.postprocess_ms,
                    total_ms=timed.total_ms,
                    provider=timed.providers[0] if timed.providers else "",
                )
            )
        if timed is None:
            raise RuntimeError("ONNX detector returned no timed result")
        return TimedDetectionsResult(
            detections=timed.detections,
            timings=timings,
            providers=timed.providers,
            input_name=timed.input_name,
            output_name=timed.output_name,
            input_shape=timed.input_shape,
            output_shape=timed.output_shape,
        )

    return run_pt, run_onnx


def collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    data: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv_version": getattr(cv2, "__version__", None),
        "pytorch_model_path": str(args.pytorch_model),
        "onnx_model_path": str(args.onnx_model),
        "pytorch_model_size_bytes": _file_size(args.pytorch_model),
        "onnx_model_size_bytes": _file_size(args.onnx_model),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "cli_args": vars(args),
    }
    try:
        import torch

        data["pytorch_version"] = torch.__version__
        data["cuda_available"] = bool(torch.cuda.is_available())
        data["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        data["pytorch_device"] = args.device
    except Exception as exc:
        data["pytorch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import ultralytics

        data["ultralytics_version"] = getattr(ultralytics, "__version__", None)
    except Exception as exc:
        data["ultralytics_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import onnxruntime as ort

        data["onnxruntime_version"] = ort.__version__
        data["onnxruntime_available_providers"] = ort.get_available_providers()
    except Exception as exc:
        data["onnxruntime_error"] = f"{type(exc).__name__}: {exc}"
        data["onnxruntime_available_providers"] = []
    return data


def write_reports(
    output_dir: Path,
    summary: dict[str, Any],
    image_results: list[ImageComparison],
    timing_rows: list[TimingSample],
    environment: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "run_config.json", vars(args))
    write_json(output_dir / "environment.json", environment)
    write_image_results_csv(output_dir / "image_results.csv", image_results)
    write_timing_csv(output_dir / "timing.csv", image_results, timing_rows)
    (output_dir / "mismatch_images.txt").write_text(format_mismatch_text(image_results), encoding="utf-8")
    (output_dir / "error_images.txt").write_text(format_error_text(image_results), encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def write_image_results_csv(path: Path, image_results: list[ImageComparison]) -> None:
    fields = [
        "image_name",
        "image_path",
        "status",
        "status_reason",
        "pytorch_detection_count",
        "onnx_detection_count",
        "matched_count",
        "pytorch_only_count",
        "onnx_only_count",
        "classes_match",
        "confidence_diff_mean",
        "confidence_diff_max",
        "bbox_iou_mean",
        "bbox_iou_min",
        "pytorch_mean_ms",
        "pytorch_median_ms",
        "pytorch_p95_ms",
        "onnx_mean_ms",
        "onnx_median_ms",
        "onnx_p95_ms",
        "speedup",
        "onnx_provider",
        "error_message",
    ]
    with path.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in image_results:
            writer.writerow(
                {
                    "image_name": result.image_name,
                    "image_path": result.image_path,
                    "status": result.status,
                    "status_reason": result.status_reason,
                    "pytorch_detection_count": _csv_value(result.pytorch_detection_count),
                    "onnx_detection_count": _csv_value(result.onnx_detection_count),
                    "matched_count": _csv_value(result.matched_count),
                    "pytorch_only_count": _csv_value(result.pytorch_only_count),
                    "onnx_only_count": _csv_value(result.onnx_only_count),
                    "classes_match": _csv_value(result.classes_match),
                    "confidence_diff_mean": _csv_value(result.confidence_diff_mean),
                    "confidence_diff_max": _csv_value(result.confidence_diff_max),
                    "bbox_iou_mean": _csv_value(result.bbox_iou_mean),
                    "bbox_iou_min": _csv_value(result.bbox_iou_min),
                    "pytorch_mean_ms": _csv_value(_stat(result.pytorch_timing, "total_ms", "mean_ms")),
                    "pytorch_median_ms": _csv_value(_stat(result.pytorch_timing, "total_ms", "median_ms")),
                    "pytorch_p95_ms": _csv_value(_stat(result.pytorch_timing, "total_ms", "p95_ms")),
                    "onnx_mean_ms": _csv_value(_stat(result.onnx_timing, "total_ms", "mean_ms")),
                    "onnx_median_ms": _csv_value(_stat(result.onnx_timing, "total_ms", "median_ms")),
                    "onnx_p95_ms": _csv_value(_stat(result.onnx_timing, "total_ms", "p95_ms")),
                    "speedup": _csv_value(result.speedup),
                    "onnx_provider": result.onnx_provider,
                    "error_message": result.error_message,
                }
            )


def write_timing_csv(path: Path, image_results: list[ImageComparison], timing_rows: list[TimingSample]) -> None:
    fields = ["image_name", "backend", "repeat_index", "preprocess_ms", "inference_ms", "postprocess_ms", "total_ms", "provider"]
    with path.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in image_results:
            for sample in result.timing_rows:
                writer.writerow(
                    {
                        "image_name": result.image_name,
                        "backend": sample.backend,
                        "repeat_index": sample.repeat_index,
                        "preprocess_ms": _csv_value(sample.preprocess_ms),
                        "inference_ms": _csv_value(sample.inference_ms),
                        "postprocess_ms": _csv_value(sample.postprocess_ms),
                        "total_ms": _csv_value(sample.total_ms),
                        "provider": sample.provider,
                    }
                )


def write_per_image_json(output_dir: Path, result: ImageComparison) -> None:
    write_json(output_dir / "per_image" / f"{Path(result.image_name).stem}.json", result)


def maybe_write_side_by_side(
    output_dir: Path,
    image_path: Path,
    result: ImageComparison,
    class_names: dict[int, str],
    save_all_images: bool,
) -> None:
    if result.status == "PASS" and not save_all_images:
        return
    target_dir = output_dir / ("comparison_images" if save_all_images else "failures")
    output_path = target_dir / f"{image_path.stem}_side_by_side.jpg"
    try:
        write_comparison_image(
            image_path,
            [dict_to_detection(item) for item in result.pytorch_detections],
            [dict_to_detection(item) for item in result.onnx_detections],
            output_path,
            class_names,
        )
    except Exception as exc:
        print(f"WARNING: failed to write side-by-side image for {image_path.name}: {exc}", file=sys.stderr)


def write_comparison_image(
    image_path: Path,
    pt_detections: list[Detection],
    onnx_detections: list[Detection],
    output_path: Path,
    class_names: dict[int, str],
) -> None:
    temp_dir = output_path.parent / "_annotated"
    pt_path = temp_dir / f"{image_path.stem}_pytorch.jpg"
    onnx_path = temp_dir / f"{image_path.stem}_onnx.jpg"
    draw_detections(image_path, pt_detections, pt_path, "PyTorch", class_names)
    draw_detections(image_path, onnx_detections, onnx_path, "ONNX Runtime", class_names)
    left = cv2.imread(str(pt_path))
    right = cv2.imread(str(onnx_path))
    if left is None or right is None:
        raise FileNotFoundError("Annotated images were not created")
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), np.hstack([left, right])):
        raise RuntimeError(f"Failed to write image: {output_path}")


def format_mismatch_text(image_results: list[ImageComparison]) -> str:
    blocks: list[str] = []
    for result in image_results:
        if result.status not in {"WARNING", "FAIL"}:
            continue
        lines = [f"[{result.status}] {result.image_name}"]
        if result.confidence_diff_max is not None:
            lines.append(f"- confidence_diff_max: {result.confidence_diff_max:.6f}")
        if result.bbox_iou_min is not None:
            lines.append(f"- bbox_iou_min: {result.bbox_iou_min:.6f}")
        if result.pytorch_only_count is not None:
            lines.append(f"- pytorch_only: {result.pytorch_only_count}")
        if result.onnx_only_count is not None:
            lines.append(f"- onnx_only: {result.onnx_only_count}")
        if result.classes_match is not None:
            lines.append(f"- classes_match: {str(result.classes_match).lower()}")
        lines.append(f"- reason: {result.status_reason}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def format_error_text(image_results: list[ImageComparison]) -> str:
    blocks = [
        f"[ERROR] {result.image_name}\n- reason: {result.status_reason}\n- error: {result.error_message}"
        for result in image_results
        if result.status == "ERROR"
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def print_startup(args: argparse.Namespace, images: list[Path], environment: dict[str, Any]) -> None:
    print("=== PyTorch vs ONNX Batch Comparison ===")
    print(f"PyTorch model: {args.pytorch_model}")
    print(f"ONNX model:    {args.onnx_model}")
    print(f"Image directory: {args.images}")
    print(f"Image count: {len(images)}")
    print(f"imgsz/conf/iou/match_iou: {args.imgsz} / {args.conf} / {args.iou} / {args.match_iou}")
    print(f"repeat/warmup: {args.repeat} / {args.warmup}")
    print(f"PyTorch device: {args.device}")
    print(f"ONNX available providers: {environment.get('onnxruntime_available_providers', [])}")
    print(f"Output directory: {args.output}")
    if "CUDAExecutionProvider" not in environment.get("onnxruntime_available_providers", []):
        print("WARNING: CUDAExecutionProvider is not available; ONNX Runtime may use CPU fallback.")


def print_progress(index: int, total: int, result: ImageComparison) -> None:
    print(f"[{index}/{total}] {result.image_name} - {result.status}")
    if result.status == "ERROR":
        print(f"ERROR: {result.error_message}")
        return
    print(
        f"PT {result.pytorch_detection_count} / ONNX {result.onnx_detection_count} / "
        f"matched {result.matched_count} / speedup {_format_float(result.speedup)}x"
    )
    if result.status != "PASS":
        print(result.status_reason)


def print_final_summary(summary: dict[str, Any], output_dir: Path) -> None:
    counts = summary["counts"]
    detections = summary["detection_summary"]
    accuracy = summary["accuracy_difference_summary"]
    timing = summary["timing_summary"]
    providers = summary["onnx_runtime"]["selected_providers"]
    print("=== PyTorch vs ONNX Batch Comparison ===")
    print(f"Total images: {counts['total_images']}")
    print(f"PASS/WARNING/FAIL/ERROR: {counts['pass_count']} / {counts['warning_count']} / {counts['fail_count']} / {counts['error_count']}")
    print(f"PyTorch detections: {detections['total_pytorch_detections']}")
    print(f"ONNX detections: {detections['total_onnx_detections']}")
    print(f"Matched detections: {detections['total_matched_detections']}")
    print(f"PyTorch only / ONNX only: {detections['total_pytorch_only']} / {detections['total_onnx_only']}")
    print(
        "Confidence diff mean/max: "
        f"{_format_float(accuracy['overall_confidence_diff_mean'])} / {_format_float(accuracy['overall_confidence_diff_max'])}"
    )
    print(f"BBox IoU mean/min: {_format_float(accuracy['overall_bbox_iou_mean'])} / {_format_float(accuracy['overall_bbox_iou_min'])}")
    print(f"PyTorch mean: {_format_float(timing['pytorch_total_mean_ms'])} ms")
    print(f"ONNX mean: {_format_float(timing['onnx_total_mean_ms'])} ms")
    print(f"Speedup: {_format_float(timing['speedup_mean'])}x")
    print(f"ONNX provider: {', '.join(providers) if providers else 'N/A'}")
    print(f"Final status: {summary['final_status']}")
    print(f"Output: {output_dir}")


def exit_code_for_status(status: str, fail_on_warning: bool) -> int:
    if status == "PASS":
        return 0
    if status == "WARNING" and not fail_on_warning:
        return 0
    return 1


def match_to_dict(match: DetectionMatch) -> dict[str, Any]:
    row: dict[str, Any] = {"status": match.status, "bbox_iou": float(match.iou)}
    if match.pt is not None:
        row["pytorch"] = detection_to_dict(match.pt)
    if match.onnx is not None:
        row["onnx"] = detection_to_dict(match.onnx)
    if match.pt is not None and match.onnx is not None:
        row["class_id"] = match.pt.class_id
        row["class_name"] = match.pt.class_name
        row["confidence_diff_abs"] = abs(match.pt.confidence - match.onnx.confidence)
    return row


def dict_to_detection(data: dict[str, Any]) -> Detection:
    bbox = data["bbox"]
    return Detection(
        class_id=int(data["class_id"]),
        class_name=str(data["class_name"]),
        confidence=float(data["confidence"]),
        x1=int(bbox[0]),
        y1=int(bbox[1]),
        x2=int(bbox[2]),
        y2=int(bbox[3]),
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    return value


def _create_output_dirs(output_dir: Path) -> None:
    for name in ("comparison_images", "failures", "per_image"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def _normalize_extension(extension: str) -> str:
    extension = extension.lower().strip()
    return extension if extension.startswith(".") else f".{extension}"


def _stat(summary: dict[str, dict[str, float | None]], group: str, key: str) -> float | None:
    value = summary.get(group, {}).get(key)
    return float(value) if value is not None else None


def _mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _count_status(results: list[ImageComparison], status: str) -> int:
    return sum(1 for result in results if result.status == status)


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _format_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_batch(args).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
