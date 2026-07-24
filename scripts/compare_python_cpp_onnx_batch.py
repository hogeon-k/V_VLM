from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.write_python_onnx_reference import load_class_names, postprocess_float
from service.onnx_detector import OnnxDetector, preprocess_image


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
PASS_BBOX_IOU_MIN = 0.999
PASS_CONFIDENCE_DIFF_MAX = 1e-5
WARNING_BBOX_IOU_MIN = 0.99
WARNING_CONFIDENCE_DIFF_MAX = 1e-3
CSV_ENCODING = "utf-8-sig"


@dataclass(frozen=True, slots=True)
class TimingSample:
    image_name: str
    backend: str
    repeat_index: int
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    provider: str


@dataclass(frozen=True, slots=True)
class TimedJsonResult:
    data: dict[str, Any]
    timings: list[TimingSample] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ImageComparison:
    image_name: str
    image_path: str
    status: str
    status_reason: str
    python_detection_count: int | None = None
    cpp_detection_count: int | None = None
    matched_count: int | None = None
    python_only_count: int | None = None
    cpp_only_count: int | None = None
    classes_match: bool | None = None
    confidence_diff_mean: float | None = None
    confidence_diff_max: float | None = None
    bbox_iou_mean: float | None = None
    bbox_iou_min: float | None = None
    python_timing: dict[str, dict[str, float | None]] = field(default_factory=dict)
    cpp_timing: dict[str, dict[str, float | None]] = field(default_factory=dict)
    speedup: float | None = None
    cpp_provider: str = ""
    error_message: str = ""
    matches: list[dict[str, Any]] = field(default_factory=list)
    python_result: dict[str, Any] = field(default_factory=dict)
    cpp_result: dict[str, Any] = field(default_factory=dict)
    timing_rows: list[TimingSample] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BatchRunResult:
    summary: dict[str, Any]
    image_results: list[ImageComparison]
    timing_rows: list[TimingSample]
    exit_code: int


PythonRunner = Callable[[Path], TimedJsonResult]
CppRunner = Callable[[Path], TimedJsonResult]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python ONNX and C++ ONNX Runtime outputs over an image batch.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("models/model_metadata.json"))
    parser.add_argument("--cpp-exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--extensions", default=",".join(SUPPORTED_EXTENSIONS))
    return parser.parse_args(argv)


def collect_images(image_dir: Path, extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS) -> list[Path]:
    normalized = tuple(_normalize_extension(extension) for extension in extensions)
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in normalized)


def validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not args.images.exists() or not args.images.is_dir():
        errors.append(f"Image directory does not exist: {args.images}")
    if not args.model.exists() or not args.model.is_file():
        errors.append(f"ONNX model file does not exist: {args.model}")
    if not args.cpp_exe.exists() or not args.cpp_exe.is_file():
        errors.append(f"C++ executable does not exist: {args.cpp_exe}")
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
    try:
        args.output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        errors.append(f"Cannot create output directory {args.output}: {exc}")
    return errors


def extensions_from_arg(value: str) -> tuple[str, ...]:
    extensions = tuple(_normalize_extension(part.strip()) for part in value.split(",") if part.strip())
    return extensions or SUPPORTED_EXTENSIONS


def bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    area_second = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    union = area_first + area_second - inter
    return 0.0 if union <= 0 else float(inter / union)


def match_detections(
    python_detections: list[dict[str, Any]],
    cpp_detections: list[dict[str, Any]],
    match_iou: float,
) -> list[dict[str, Any]]:
    pairs: list[tuple[float, int, int]] = []
    for python_index, python_det in enumerate(python_detections):
        for cpp_index, cpp_det in enumerate(cpp_detections):
            if int(python_det["class_id"]) != int(cpp_det["class_id"]):
                continue
            iou = bbox_iou(python_det["bbox"], cpp_det["bbox"])
            if iou >= match_iou:
                pairs.append((iou, python_index, cpp_index))

    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_python: set[int] = set()
    used_cpp: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, python_index, cpp_index in pairs:
        if python_index in used_python or cpp_index in used_cpp:
            continue
        python_det = python_detections[python_index]
        cpp_det = cpp_detections[cpp_index]
        used_python.add(python_index)
        used_cpp.add(cpp_index)
        matches.append(
            {
                "status": "MATCHED",
                "python_index": python_index,
                "cpp_index": cpp_index,
                "class_id": int(python_det["class_id"]),
                "class_name": str(python_det.get("class_name", python_det["class_id"])),
                "class_match": int(python_det["class_id"]) == int(cpp_det["class_id"]),
                "confidence_diff_abs": abs(float(python_det["confidence"]) - float(cpp_det["confidence"])),
                "bbox_iou": float(iou),
                "python": python_det,
                "cpp": cpp_det,
            }
        )

    for python_index, python_det in enumerate(python_detections):
        if python_index not in used_python:
            matches.append({"status": "PYTHON_ONLY", "python_index": python_index, "python": python_det, "bbox_iou": 0.0})
    for cpp_index, cpp_det in enumerate(cpp_detections):
        if cpp_index not in used_cpp:
            matches.append({"status": "CPP_ONLY", "cpp_index": cpp_index, "cpp": cpp_det, "bbox_iou": 0.0})
    return sorted(matches, key=lambda item: (item["status"] != "MATCHED", item.get("python_index", 10**9), item.get("cpp_index", 10**9)))


def judge_status(
    python_count: int,
    cpp_count: int,
    matched_count: int,
    python_only_count: int,
    cpp_only_count: int,
    classes_match: bool,
    confidence_diff_max: float,
    bbox_iou_min: float,
) -> tuple[str, str]:
    if python_count == 0 and cpp_count == 0:
        return "PASS", "Both backends returned zero detections."
    if python_only_count > 0 or cpp_only_count > 0 or matched_count != python_count or matched_count != cpp_count or not classes_match:
        reasons = []
        if python_count != cpp_count:
            reasons.append(f"detection_count={python_count}/{cpp_count}")
        if python_only_count > 0:
            reasons.append(f"python_only_count={python_only_count}")
        if cpp_only_count > 0:
            reasons.append(f"cpp_only_count={cpp_only_count}")
        if not classes_match:
            reasons.append("classes_match=false")
        return "FAIL", "; ".join(reasons)
    if bbox_iou_min < WARNING_BBOX_IOU_MIN:
        return "FAIL", f"bbox_iou_min={bbox_iou_min:.9f} < {WARNING_BBOX_IOU_MIN}"
    if bbox_iou_min >= PASS_BBOX_IOU_MIN and confidence_diff_max <= PASS_CONFIDENCE_DIFF_MAX:
        return "PASS", "Detection counts, classes, confidence, and boxes are within PASS thresholds."
    if confidence_diff_max <= WARNING_CONFIDENCE_DIFF_MAX:
        return "WARNING", "Detection counts/classes match, but precision is outside PASS thresholds."
    return "FAIL", f"confidence_diff_max={confidence_diff_max:.9g} > {WARNING_CONFIDENCE_DIFF_MAX}"


def compare_results(image_path: Path, python_result: TimedJsonResult, cpp_result: TimedJsonResult, match_iou: float) -> ImageComparison:
    python_detections = list(python_result.data.get("detections", []))
    cpp_detections = list(cpp_result.data.get("detections", []))
    matches = match_detections(python_detections, cpp_detections, match_iou)
    matched = [match for match in matches if match["status"] == "MATCHED"]
    python_only = [match for match in matches if match["status"] == "PYTHON_ONLY"]
    cpp_only = [match for match in matches if match["status"] == "CPP_ONLY"]
    conf_diffs = [float(match["confidence_diff_abs"]) for match in matched]
    bbox_ious = [float(match["bbox_iou"]) for match in matched]
    classes_match = all(bool(match.get("class_match", False)) for match in matched)
    confidence_diff_mean = float(statistics.fmean(conf_diffs)) if conf_diffs else 0.0
    confidence_diff_max = float(max(conf_diffs)) if conf_diffs else 0.0
    bbox_iou_mean = float(statistics.fmean(bbox_ious)) if bbox_ious else 0.0
    bbox_iou_min = float(min(bbox_ious)) if bbox_ious else 1.0
    status, reason = judge_status(
        len(python_detections),
        len(cpp_detections),
        len(matched),
        len(python_only),
        len(cpp_only),
        classes_match,
        confidence_diff_max,
        bbox_iou_min,
    )
    python_timing = summarize_backend_timings(python_result.timings)
    cpp_timing = summarize_backend_timings(cpp_result.timings)
    speedup = calculate_speedup(_stat(python_timing, "total_ms", "mean_ms"), _stat(cpp_timing, "total_ms", "mean_ms"))
    return ImageComparison(
        image_name=image_path.name,
        image_path=str(image_path),
        status=status,
        status_reason=reason,
        python_detection_count=len(python_detections),
        cpp_detection_count=len(cpp_detections),
        matched_count=len(matched),
        python_only_count=len(python_only),
        cpp_only_count=len(cpp_only),
        classes_match=classes_match,
        confidence_diff_mean=confidence_diff_mean,
        confidence_diff_max=confidence_diff_max,
        bbox_iou_mean=bbox_iou_mean,
        bbox_iou_min=bbox_iou_min,
        python_timing=python_timing,
        cpp_timing=cpp_timing,
        speedup=speedup,
        cpp_provider=str(cpp_result.data.get("provider", "")),
        matches=matches,
        python_result=python_result.data,
        cpp_result=cpp_result.data,
        timing_rows=[*python_result.timings, *cpp_result.timings],
    )


def error_comparison(image_path: Path, exc: Exception) -> ImageComparison:
    return ImageComparison(
        image_name=image_path.name,
        image_path=str(image_path),
        status="ERROR",
        status_reason="Execution or comparison failed.",
        error_message=f"{type(exc).__name__}: {exc}",
    )


def run_python_reference_factory(args: argparse.Namespace) -> PythonRunner:
    class_names = load_class_names(args.metadata)
    detector = OnnxDetector(
        args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        class_names=class_names,
        requested_provider="CPUExecutionProvider",
        preload_torch_cuda=False,
    )
    session = detector._load_session()

    def run(image_path: Path) -> TimedJsonResult:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Input image not found or unreadable: {image_path}")
        timings: list[TimingSample] = []
        last_data: dict[str, Any] | None = None
        for index in range(args.warmup + args.repeat):
            total_start = _perf_counter_ms()
            start = _perf_counter_ms()
            tensor, letterbox_info = preprocess_image(image, args.imgsz)
            preprocess_ms = _perf_counter_ms() - start
            start = _perf_counter_ms()
            outputs = session.run([detector._output_name], {detector._input_name: tensor})
            inference_ms = _perf_counter_ms() - start
            start = _perf_counter_ms()
            output = np.asarray(outputs[0])
            detections = postprocess_float(output, letterbox_info, args.conf, args.iou, class_names)
            postprocess_ms = _perf_counter_ms() - start
            total_ms = _perf_counter_ms() - total_start
            data = {
                "model": str(args.model),
                "image": str(image_path),
                "provider": detector.actual_providers[0] if detector.actual_providers else "",
                "input_name": detector._input_name,
                "output_name": detector._output_name,
                "input_shape": detector._input_shape,
                "output_shape": list(output.shape),
                "config": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou},
                "timing_ms": {
                    "preprocess": preprocess_ms,
                    "inference": inference_ms,
                    "postprocess": postprocess_ms,
                    "total": total_ms,
                },
                "detections": detections,
            }
            last_data = data
            if index >= args.warmup:
                timings.append(_timing_sample(image_path.name, "python", index - args.warmup, data))
        if last_data is None:
            raise RuntimeError("No Python ONNX inference run was executed.")
        return TimedJsonResult(last_data, timings)

    return run


def run_cpp_reference_factory(args: argparse.Namespace) -> CppRunner:
    temp_root = args.output / "_cpp_runs"
    if temp_root.exists():
        shutil.rmtree(temp_root)

    def run(image_path: Path) -> TimedJsonResult:
        timings: list[TimingSample] = []
        last_data: dict[str, Any] | None = None
        image_temp_root = temp_root / image_path.stem
        for index in range(args.warmup + args.repeat):
            run_dir = image_temp_root / f"run_{index:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(args.cpp_exe),
                "--model",
                str(args.model),
                "--metadata",
                str(args.metadata),
                "--image",
                str(image_path),
                "--output",
                str(run_dir),
                "--imgsz",
                str(args.imgsz),
                "--conf",
                str(args.conf),
                "--iou",
                str(args.iou),
            ]
            completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"C++ inference failed for {image_path.name}: {completed.stderr.strip() or completed.stdout.strip()}")
            result_path = run_dir / "result.json"
            if not result_path.exists():
                raise FileNotFoundError(f"C++ result JSON was not written: {result_path}")
            data = json.loads(result_path.read_text(encoding="utf-8"))
            last_data = data
            if index >= args.warmup:
                timings.append(_timing_sample(image_path.name, "cpp", index - args.warmup, data))
            shutil.rmtree(run_dir, ignore_errors=True)
        if last_data is None:
            raise RuntimeError("No C++ ONNX inference run was executed.")
        shutil.rmtree(image_temp_root, ignore_errors=True)
        return TimedJsonResult(last_data, timings)

    return run


def run_batch(
    args: argparse.Namespace,
    python_runner: PythonRunner | None = None,
    cpp_runner: CppRunner | None = None,
) -> BatchRunResult:
    errors = validate_args(args)
    if errors:
        args.output.mkdir(parents=True, exist_ok=True)
        summary = {"final_status": "ERROR", "errors": errors}
        write_json(args.output / "summary.json", summary)
        return BatchRunResult(summary, [], [], 2)

    images = collect_images(args.images, extensions_from_arg(args.extensions))
    if not images:
        summary = {"final_status": "ERROR", "errors": [f"No supported images found in: {args.images}"], "image_count": 0}
        write_json(args.output / "summary.json", summary)
        write_per_image_csv(args.output / "per_image.csv", [])
        return BatchRunResult(summary, [], [], 2)

    _create_output_dirs(args.output)
    python_runner = python_runner or run_python_reference_factory(args)
    cpp_runner = cpp_runner or run_cpp_reference_factory(args)
    results: list[ImageComparison] = []
    timing_rows: list[TimingSample] = []

    print_startup(args, images)
    for index, image_path in enumerate(images, start=1):
        try:
            python_result = python_runner(image_path)
            cpp_result = cpp_runner(image_path)
            result = compare_results(image_path, python_result, cpp_result, args.match_iou)
            write_json(args.output / "python" / f"{image_path.stem}.json", python_result.data)
            write_json(args.output / "cpp" / f"{image_path.stem}.json", cpp_result.data)
        except Exception as exc:
            result = error_comparison(image_path, exc)
        results.append(result)
        timing_rows.extend(result.timing_rows)
        write_json(args.output / "comparisons" / f"{image_path.stem}.json", result)
        if should_copy_failure_case(result):
            copy_failure_case(image_path, args.output / "failure_cases")
        print_progress(index, len(images), result)

    shutil.rmtree(args.output / "_cpp_runs", ignore_errors=True)
    summary = build_summary(args, images, results, timing_rows)
    write_json(args.output / "summary.json", summary)
    write_per_image_csv(args.output / "per_image.csv", results)
    print_final_summary(summary, args.output)
    return BatchRunResult(summary, results, timing_rows, exit_code_for_status(summary["final_status"]))


def build_summary(args: argparse.Namespace, images: list[Path], results: list[ImageComparison], timing_rows: list[TimingSample]) -> dict[str, Any]:
    processed = [result for result in results if result.status != "ERROR"]
    matched_conf_diffs = [
        float(match["confidence_diff_abs"])
        for result in processed
        for match in result.matches
        if match.get("status") == "MATCHED"
    ]
    matched_ious = [
        float(match["bbox_iou"])
        for result in processed
        for match in result.matches
        if match.get("status") == "MATCHED"
    ]
    python_timing = summarize_timing_rows(timing_rows, "python")
    cpp_timing = summarize_timing_rows(timing_rows, "cpp")
    final = final_status(results)
    return {
        "final_status": final,
        "image_count": len(images),
        "counts": {
            "total_images": len(images),
            "pass_count": _count_status(results, "PASS"),
            "warning_count": _count_status(results, "WARNING"),
            "fail_count": _count_status(results, "FAIL"),
            "error_count": _count_status(results, "ERROR"),
        },
        "detection_summary": {
            "python_detection_count": sum(result.python_detection_count or 0 for result in processed),
            "cpp_detection_count": sum(result.cpp_detection_count or 0 for result in processed),
            "matched_detection_count": sum(result.matched_count or 0 for result in processed),
            "python_only_count": sum(result.python_only_count or 0 for result in processed),
            "cpp_only_count": sum(result.cpp_only_count or 0 for result in processed),
        },
        "accuracy_summary": {
            "confidence_diff_mean": _mean_or_none(matched_conf_diffs),
            "confidence_diff_max": max(matched_conf_diffs) if matched_conf_diffs else None,
            "bbox_iou_mean": _mean_or_none(matched_ious),
            "bbox_iou_min": min(matched_ious) if matched_ious else None,
        },
        "timing_summary": {
            "python": python_timing,
            "cpp": cpp_timing,
            "speedup": {
                "total_mean": calculate_speedup(_stat(python_timing, "total_ms", "mean_ms"), _stat(cpp_timing, "total_ms", "mean_ms")),
                "total_median": calculate_speedup(_stat(python_timing, "total_ms", "median_ms"), _stat(cpp_timing, "total_ms", "median_ms")),
                "total_p95": calculate_speedup(_stat(python_timing, "total_ms", "p95_ms"), _stat(cpp_timing, "total_ms", "p95_ms")),
                "inference_mean": calculate_speedup(_stat(python_timing, "inference_ms", "mean_ms"), _stat(cpp_timing, "inference_ms", "mean_ms")),
            },
        },
        "thresholds": {
            "match_iou": args.match_iou,
            "pass_bbox_iou_min": PASS_BBOX_IOU_MIN,
            "pass_confidence_diff_max": PASS_CONFIDENCE_DIFF_MAX,
            "warning_bbox_iou_min": WARNING_BBOX_IOU_MIN,
            "warning_confidence_diff_max": WARNING_CONFIDENCE_DIFF_MAX,
        },
        "run_config": {
            "images": str(args.images),
            "model": str(args.model),
            "cpp_exe": str(args.cpp_exe),
            "output": str(args.output),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
        },
        "per_image": [to_jsonable(result) for result in results],
    }


def timing_stats(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None, "fps": None}
    values = sorted(float(value) for value in samples)
    mean_ms = float(statistics.fmean(values))
    return {
        "count": len(values),
        "mean_ms": mean_ms,
        "median_ms": float(statistics.median(values)),
        "p95_ms": percentile(values, 95.0),
        "min_ms": float(values[0]),
        "max_ms": float(values[-1]),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else None,
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
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (rank - lower))


def summarize_backend_timings(timings: list[TimingSample]) -> dict[str, dict[str, float | None]]:
    return {
        field_name: timing_stats([float(getattr(sample, field_name)) for sample in timings])
        for field_name in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms")
    }


def summarize_timing_rows(timing_rows: list[TimingSample], backend: str) -> dict[str, dict[str, float | None]]:
    return summarize_backend_timings([sample for sample in timing_rows if sample.backend == backend])


def write_per_image_csv(path: Path, results: list[ImageComparison]) -> None:
    fields = [
        "image_name",
        "status",
        "status_reason",
        "python_detection_count",
        "cpp_detection_count",
        "matched_count",
        "python_only_count",
        "cpp_only_count",
        "classes_match",
        "confidence_diff_mean",
        "confidence_diff_max",
        "bbox_iou_mean",
        "bbox_iou_min",
        "python_total_mean_ms",
        "cpp_total_mean_ms",
        "speedup",
        "cpp_provider",
        "error_message",
    ]
    with path.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "image_name": result.image_name,
                    "status": result.status,
                    "status_reason": result.status_reason,
                    "python_detection_count": _csv_value(result.python_detection_count),
                    "cpp_detection_count": _csv_value(result.cpp_detection_count),
                    "matched_count": _csv_value(result.matched_count),
                    "python_only_count": _csv_value(result.python_only_count),
                    "cpp_only_count": _csv_value(result.cpp_only_count),
                    "classes_match": _csv_value(result.classes_match),
                    "confidence_diff_mean": _csv_value(result.confidence_diff_mean),
                    "confidence_diff_max": _csv_value(result.confidence_diff_max),
                    "bbox_iou_mean": _csv_value(result.bbox_iou_mean),
                    "bbox_iou_min": _csv_value(result.bbox_iou_min),
                    "python_total_mean_ms": _csv_value(_stat(result.python_timing, "total_ms", "mean_ms")),
                    "cpp_total_mean_ms": _csv_value(_stat(result.cpp_timing, "total_ms", "mean_ms")),
                    "speedup": _csv_value(result.speedup),
                    "cpp_provider": result.cpp_provider,
                    "error_message": result.error_message,
                }
            )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def should_copy_failure_case(result: ImageComparison) -> bool:
    if result.status in {"FAIL", "ERROR"}:
        return True
    if result.status == "WARNING":
        return True
    return False


def copy_failure_case(image_path: Path, failure_dir: Path) -> None:
    failure_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, failure_dir / image_path.name)


def calculate_speedup(python_ms: float | None, cpp_ms: float | None) -> float | None:
    if python_ms is None or cpp_ms is None or cpp_ms <= 0:
        return None
    return float(python_ms / cpp_ms)


def final_status(results: list[ImageComparison]) -> str:
    statuses = {result.status for result in results}
    if "ERROR" in statuses:
        return "ERROR"
    if "FAIL" in statuses:
        return "FAIL"
    if "WARNING" in statuses:
        return "WARNING"
    return "PASS"


def exit_code_for_status(status: str) -> int:
    if status == "PASS":
        return 0
    if status == "WARNING":
        return 1
    return 2


def print_startup(args: argparse.Namespace, images: list[Path]) -> None:
    print("=== Python ONNX vs C++ ONNX Batch Comparison ===")
    print(f"ONNX model: {args.model}")
    print(f"C++ exe:    {args.cpp_exe}")
    print(f"Images: {args.images}")
    print(f"Image count: {len(images)}")
    print(f"imgsz/conf/iou/match_iou: {args.imgsz} / {args.conf} / {args.iou} / {args.match_iou}")
    print(f"warmup/repeat: {args.warmup} / {args.repeat}")
    print(f"Output: {args.output}")


def print_progress(index: int, total: int, result: ImageComparison) -> None:
    print(f"[{index}/{total}] {result.image_name} - {result.status}")
    if result.status == "ERROR":
        print(f"ERROR: {result.error_message}")
        return
    print(
        f"Python {result.python_detection_count} / C++ {result.cpp_detection_count} / "
        f"matched {result.matched_count} / speedup {_format_float(result.speedup)}x"
    )
    if result.status != "PASS":
        print(result.status_reason)


def print_final_summary(summary: dict[str, Any], output_dir: Path) -> None:
    counts = summary["counts"]
    detections = summary["detection_summary"]
    accuracy = summary["accuracy_summary"]
    speedup = summary["timing_summary"]["speedup"]
    print("=== Batch Summary ===")
    print(f"Total images: {counts['total_images']}")
    print(f"PASS/WARNING/FAIL/ERROR: {counts['pass_count']} / {counts['warning_count']} / {counts['fail_count']} / {counts['error_count']}")
    print(f"Python/C++ detections: {detections['python_detection_count']} / {detections['cpp_detection_count']}")
    print(f"Matched/Python only/C++ only: {detections['matched_detection_count']} / {detections['python_only_count']} / {detections['cpp_only_count']}")
    print(f"Confidence diff mean/max: {_format_float(accuracy['confidence_diff_mean'])} / {_format_float(accuracy['confidence_diff_max'])}")
    print(f"BBox IoU mean/min: {_format_float(accuracy['bbox_iou_mean'])} / {_format_float(accuracy['bbox_iou_min'])}")
    print(f"Speedup total mean: {_format_float(speedup['total_mean'])}x")
    print(f"Final status: {summary['final_status']}")
    print(f"Output: {output_dir}")


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
    for name in ("python", "cpp", "comparisons", "failure_cases"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "failure_cases" / ".gitkeep").touch()


def _timing_sample(image_name: str, backend: str, repeat_index: int, data: dict[str, Any]) -> TimingSample:
    timing = data.get("timing_ms", {})
    return TimingSample(
        image_name=image_name,
        backend=backend,
        repeat_index=repeat_index,
        preprocess_ms=float(timing.get("preprocess", 0.0)),
        inference_ms=float(timing.get("inference", 0.0)),
        postprocess_ms=float(timing.get("postprocess", 0.0)),
        total_ms=float(timing.get("total", 0.0)),
        provider=str(data.get("provider", "")),
    )


def _normalize_extension(extension: str) -> str:
    extension = extension.lower().strip()
    return extension if extension.startswith(".") else f".{extension}"


def _perf_counter_ms() -> float:
    import time

    return time.perf_counter() * 1000.0


def _stat(summary: dict[str, dict[str, float | None]], group: str, key: str) -> float | None:
    value = summary.get(group, {}).get(key)
    return float(value) if value is not None else None


def _mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _count_status(results: list[ImageComparison], status: str) -> int:
    return sum(1 for result in results if result.status == status)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _format_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6g}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_batch(args).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
