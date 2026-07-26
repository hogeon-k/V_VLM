from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass
class ImageResult:
    image_name: str
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float | None = None
    end_to_end_ms: float | None = None
    status: str = "PASS"
    error: str = ""


@dataclass
class BackendResult:
    key: str
    backend: str
    precision: str
    path: Path
    summary: dict[str, Any]
    images: dict[str, ImageResult]
    inference_mean_ms: float | None = None
    inference_median_ms: float | None = None
    inference_p95_ms: float | None = None
    end_to_end_mean_ms: float | None = None
    end_to_end_median_ms: float | None = None
    end_to_end_p95_ms: float | None = None
    image_count: int = 0
    total_detection_count: int = 0
    failed_image_count: int = 0
    validation_mismatch_count: int = 0
    artifact_path: Path | None = None
    artifact_size_bytes: int | None = None


@dataclass
class PairComparison:
    reference_backend: str
    candidate_backend: str
    image_name: str
    reference_count: int
    candidate_count: int
    matched_count: int
    unmatched_reference_count: int
    unmatched_candidate_count: int
    class_mismatch_count: int
    confidence_diff_max: float
    confidence_diff_mean: float
    bbox_diff_max: float
    bbox_iou_min: float
    bbox_iou_mean: float
    threshold_boundary_unmatched_reference_count: int
    threshold_boundary_unmatched_candidate_count: int
    structural_unmatched_reference_count: int
    structural_unmatched_candidate_count: int
    status: str
    reason: str


def is_threshold_boundary_detection(
    detection: Detection,
    confidence_threshold: float,
    threshold_boundary_margin: float,
) -> bool:
    return abs(detection.confidence - confidence_threshold) <= threshold_boundary_margin


def boundary_distance(
    detection: Detection,
    confidence_threshold: float,
) -> float:
    return abs(detection.confidence - confidence_threshold)


def require_file(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_json(path: Path) -> Any:
    require_file(path, "JSON file")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    require_file(path, "CSV file")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def as_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer: {value!r}") from exc


def normalize_bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"{label} bbox must be [x1, y1, x2, y2].")
    x1, y1, x2, y2 = (as_float(item, f"{label}.bbox") for item in value)
    if x2 < x1 or y2 < y1:
        raise ValueError(f"{label} bbox has invalid coordinate order: {value!r}")
    return (x1, y1, x2, y2)


def parse_detection(data: dict[str, Any], label: str) -> Detection:
    if "class_id" not in data:
        raise ValueError(f"{label} missing class_id.")
    if "confidence" not in data:
        raise ValueError(f"{label} missing confidence.")
    return Detection(
        class_id=as_int(data["class_id"], f"{label}.class_id"),
        class_name=str(data.get("class_name", "")),
        confidence=as_float(data["confidence"], f"{label}.confidence"),
        bbox=normalize_bbox(data.get("bbox"), label),
    )


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(math.ceil(p * len(ordered)) - 1, len(ordered) - 1))
    return ordered[index]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def qps(mean_ms: float | None) -> float | None:
    if mean_ms is None or mean_ms <= 0.0:
        return None
    return 1000.0 / mean_ms


def bbox_iou(first: Detection, second: Detection) -> float:
    ax1, ay1, ax2, ay2 = first.bbox
    bx1, by1, bx2, by2 = second.bbox
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else inter / denom


def bbox_max_abs_diff(first: Detection, second: Detection) -> float:
    return max(abs(left - right) for left, right in zip(first.bbox, second.bbox))


def match_detections(
    reference: list[Detection],
    candidate: list[Detection],
    match_iou: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int], int]:
    pairs: list[tuple[float, int, int]] = []
    cross_class_pairs: list[tuple[float, int, int]] = []
    for ref_index, ref in enumerate(reference):
        for cand_index, cand in enumerate(candidate):
            iou = bbox_iou(ref, cand)
            if ref.class_id == cand.class_id:
                if iou >= match_iou:
                    pairs.append((iou, ref_index, cand_index))
            elif iou >= match_iou:
                cross_class_pairs.append((iou, ref_index, cand_index))

    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_ref: set[int] = set()
    used_cand: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for iou, ref_index, cand_index in pairs:
        if ref_index in used_ref or cand_index in used_cand:
            continue
        used_ref.add(ref_index)
        used_cand.add(cand_index)
        matched.append((ref_index, cand_index, iou))

    class_mismatch_count = 0
    for _iou, ref_index, cand_index in sorted(cross_class_pairs, key=lambda item: (-item[0], item[1], item[2])):
        if ref_index in used_ref or cand_index in used_cand:
            continue
        used_ref.add(ref_index)
        used_cand.add(cand_index)
        class_mismatch_count += 1

    unmatched_ref = [index for index in range(len(reference)) if index not in used_ref]
    unmatched_cand = [index for index in range(len(candidate)) if index not in used_cand]
    return matched, unmatched_ref, unmatched_cand, class_mismatch_count


def compare_image(
    image_name: str,
    reference_backend: str,
    candidate_backend: str,
    reference: list[Detection],
    candidate: list[Detection],
    match_iou: float,
    strict_confidence_tolerance: float,
    practical_confidence_tolerance: float,
    bbox_tolerance: float,
    confidence_threshold: float,
    threshold_boundary_margin: float,
) -> tuple[PairComparison, list[dict[str, Any]]]:
    matched, unmatched_ref, unmatched_cand, class_mismatch_count = match_detections(reference, candidate, match_iou)
    conf_diffs = [abs(reference[r].confidence - candidate[c].confidence) for r, c, _ in matched]
    bbox_diffs = [bbox_max_abs_diff(reference[r], candidate[c]) for r, c, _ in matched]
    ious = [iou for _, _, iou in matched]

    confidence_diff_max = max(conf_diffs, default=0.0)
    confidence_diff_mean = mean(conf_diffs) or 0.0
    bbox_diff_max = max(bbox_diffs, default=0.0)
    bbox_iou_min = min(ious, default=1.0)
    bbox_iou_mean = mean(ious) or 1.0

    boundary_unmatched_ref = [
        index for index in unmatched_ref
        if is_threshold_boundary_detection(reference[index], confidence_threshold, threshold_boundary_margin)
    ]
    boundary_unmatched_cand = [
        index for index in unmatched_cand
        if is_threshold_boundary_detection(candidate[index], confidence_threshold, threshold_boundary_margin)
    ]
    structural_unmatched_ref = [index for index in unmatched_ref if index not in set(boundary_unmatched_ref)]
    structural_unmatched_cand = [index for index in unmatched_cand if index not in set(boundary_unmatched_cand)]

    actual_structural_failure = (
        len(reference) != len(candidate)
        and (len(structural_unmatched_ref) > 0 or len(structural_unmatched_cand) > 0)
    ) or (
        len(structural_unmatched_ref) > 0
        or len(structural_unmatched_cand) > 0
        or class_mismatch_count > 0
    )
    has_threshold_boundary_warning = bool(boundary_unmatched_ref or boundary_unmatched_cand)
    if actual_structural_failure or confidence_diff_max > practical_confidence_tolerance or bbox_diff_max > bbox_tolerance:
        status = "FAIL"
    elif confidence_diff_max > strict_confidence_tolerance or has_threshold_boundary_warning:
        status = "NUMERICAL_WARNING"
    else:
        status = "PASS"

    reasons: list[str] = []
    if structural_unmatched_ref or structural_unmatched_cand:
        if len(reference) != len(candidate):
            reasons.append("detection_count_mismatch")
    if structural_unmatched_ref:
        reasons.append("unmatched_reference")
    if structural_unmatched_cand:
        reasons.append("unmatched_candidate")
    if boundary_unmatched_ref:
        reasons.append("unmatched_reference_at_confidence_threshold_boundary")
    if boundary_unmatched_cand:
        reasons.append("unmatched_candidate_at_confidence_threshold_boundary")
    if class_mismatch_count:
        reasons.append("class_mismatch")
    if confidence_diff_max > practical_confidence_tolerance:
        reasons.append("confidence_practical_tolerance_exceeded")
    elif confidence_diff_max > strict_confidence_tolerance:
        reasons.append("confidence_strict_tolerance_exceeded")
    if bbox_diff_max > bbox_tolerance:
        reasons.append("bbox_tolerance_exceeded")
    reason = "; ".join(reasons) if reasons else "within tolerances"

    rows: list[dict[str, Any]] = []
    for ref_index, cand_index, iou in matched:
        ref = reference[ref_index]
        cand = candidate[cand_index]
        rows.append(
            {
                "image_name": image_name,
                "reference_backend": reference_backend,
                "candidate_backend": candidate_backend,
                "reference_index": ref_index,
                "candidate_index": cand_index,
                "class_id": ref.class_id,
                "class_name": ref.class_name,
                "reference_confidence": ref.confidence,
                "candidate_confidence": cand.confidence,
                "confidence_diff": abs(ref.confidence - cand.confidence),
                "reference_x1": ref.bbox[0],
                "reference_y1": ref.bbox[1],
                "reference_x2": ref.bbox[2],
                "reference_y2": ref.bbox[3],
                "candidate_x1": cand.bbox[0],
                "candidate_y1": cand.bbox[1],
                "candidate_x2": cand.bbox[2],
                "candidate_y2": cand.bbox[3],
                "bbox_max_abs_diff": bbox_max_abs_diff(ref, cand),
                "iou": iou,
                "confidence_threshold": confidence_threshold,
                "threshold_boundary_margin": threshold_boundary_margin,
                "boundary_distance": "",
                "is_threshold_boundary": "false",
                "status": "PASS"
                if abs(ref.confidence - cand.confidence) <= strict_confidence_tolerance
                and bbox_max_abs_diff(ref, cand) <= bbox_tolerance
                else "NUMERICAL_WARNING",
            }
        )
    for ref_index in unmatched_ref:
        ref = reference[ref_index]
        is_boundary = ref_index in boundary_unmatched_ref
        rows.append(
            {
                "image_name": image_name,
                "reference_backend": reference_backend,
                "candidate_backend": candidate_backend,
                "reference_index": ref_index,
                "candidate_index": "",
                "class_id": ref.class_id,
                "class_name": ref.class_name,
                "reference_confidence": ref.confidence,
                "candidate_confidence": "",
                "confidence_diff": "",
                "reference_x1": ref.bbox[0],
                "reference_y1": ref.bbox[1],
                "reference_x2": ref.bbox[2],
                "reference_y2": ref.bbox[3],
                "candidate_x1": "",
                "candidate_y1": "",
                "candidate_x2": "",
                "candidate_y2": "",
                "bbox_max_abs_diff": "",
                "iou": "",
                "confidence_threshold": confidence_threshold,
                "threshold_boundary_margin": threshold_boundary_margin,
                "boundary_distance": boundary_distance(ref, confidence_threshold),
                "is_threshold_boundary": str(is_boundary).lower(),
                "status": "WARNING_UNMATCHED_REFERENCE_THRESHOLD_BOUNDARY"
                if is_boundary else "FAIL_UNMATCHED_REFERENCE",
            }
        )
    for cand_index in unmatched_cand:
        cand = candidate[cand_index]
        is_boundary = cand_index in boundary_unmatched_cand
        rows.append(
            {
                "image_name": image_name,
                "reference_backend": reference_backend,
                "candidate_backend": candidate_backend,
                "reference_index": "",
                "candidate_index": cand_index,
                "class_id": cand.class_id,
                "class_name": cand.class_name,
                "reference_confidence": "",
                "candidate_confidence": cand.confidence,
                "confidence_diff": "",
                "reference_x1": "",
                "reference_y1": "",
                "reference_x2": "",
                "reference_y2": "",
                "candidate_x1": cand.bbox[0],
                "candidate_y1": cand.bbox[1],
                "candidate_x2": cand.bbox[2],
                "candidate_y2": cand.bbox[3],
                "bbox_max_abs_diff": "",
                "iou": "",
                "confidence_threshold": confidence_threshold,
                "threshold_boundary_margin": threshold_boundary_margin,
                "boundary_distance": boundary_distance(cand, confidence_threshold),
                "is_threshold_boundary": str(is_boundary).lower(),
                "status": "WARNING_UNMATCHED_CANDIDATE_THRESHOLD_BOUNDARY"
                if is_boundary else "FAIL_UNMATCHED_CANDIDATE",
            }
        )

    return (
        PairComparison(
            reference_backend=reference_backend,
            candidate_backend=candidate_backend,
            image_name=image_name,
            reference_count=len(reference),
            candidate_count=len(candidate),
            matched_count=len(matched),
            unmatched_reference_count=len(unmatched_ref),
            unmatched_candidate_count=len(unmatched_cand),
            class_mismatch_count=class_mismatch_count,
            confidence_diff_max=confidence_diff_max,
            confidence_diff_mean=confidence_diff_mean,
            bbox_diff_max=bbox_diff_max,
            bbox_iou_min=bbox_iou_min,
            bbox_iou_mean=bbox_iou_mean,
            threshold_boundary_unmatched_reference_count=len(boundary_unmatched_ref),
            threshold_boundary_unmatched_candidate_count=len(boundary_unmatched_cand),
            structural_unmatched_reference_count=len(structural_unmatched_ref),
            structural_unmatched_candidate_count=len(structural_unmatched_cand),
            status=status,
            reason=reason,
        ),
        rows,
    )


def load_onnx_results(path: Path) -> BackendResult:
    require_dir(path, "ONNX result directory")
    summary = load_json(path / "summary.json")
    image_rows = read_csv_rows(path / "image_results.csv")
    timing_rows = read_csv_rows(path / "timing_runs.csv") if (path / "timing_runs.csv").exists() else []
    prediction_dir = require_dir(path / "cuda" / "predictions", "ONNX CUDA prediction directory")

    images: dict[str, ImageResult] = {}
    seen: set[str] = set()
    for row in image_rows:
        image_name = row.get("image", "")
        if not image_name:
            raise ValueError("image_results.csv has a row without image.")
        if image_name in seen:
            raise ValueError(f"Duplicate image_name in ONNX image_results.csv: {image_name}")
        seen.add(image_name)
        prediction = load_json(prediction_dir / f"{Path(image_name).stem}.json")
        detections = [
            parse_detection(item, f"ONNX CUDA {image_name} detection {index}")
            for index, item in enumerate(prediction.get("detections", []))
        ]
        images[image_name] = ImageResult(
            image_name=image_name,
            detections=detections,
            inference_ms=as_float(row.get("cuda_session_mean_ms"), f"{image_name}.cuda_session_mean_ms"),
            end_to_end_ms=as_float(row.get("cuda_total_mean_ms"), f"{image_name}.cuda_total_mean_ms"),
            status=row.get("status", "PASS"),
        )

    cuda_session_runs = [
        as_float(row["session_run_ms"], "timing_runs.session_run_ms")
        for row in timing_rows
        if row.get("provider") == "cuda"
    ]
    cuda_total_runs = [
        as_float(row["total_ms"], "timing_runs.total_ms")
        for row in timing_rows
        if row.get("provider") == "cuda"
    ]
    timing = summary.get("timing", {}).get("cuda", {})
    config = summary.get("config", {})
    validation = summary.get("validation", {})
    artifact = Path(config.get("model", "models/best.onnx"))

    return BackendResult(
        key="onnx_cuda",
        backend="ONNX Runtime CUDA",
        precision="FP32",
        path=path,
        summary=summary,
        images=images,
        inference_mean_ms=timing.get("session_mean") if timing.get("session_mean") is not None else mean(cuda_session_runs),
        inference_median_ms=timing.get("session_median") if timing.get("session_median") is not None else median(cuda_session_runs),
        inference_p95_ms=timing.get("session_p95") if timing.get("session_p95") is not None else percentile(cuda_session_runs, 0.95),
        end_to_end_mean_ms=timing.get("total_mean") if timing.get("total_mean") is not None else mean(cuda_total_runs),
        end_to_end_median_ms=timing.get("total_median") if timing.get("total_median") is not None else median(cuda_total_runs),
        end_to_end_p95_ms=percentile(cuda_total_runs, 0.95),
        image_count=as_int(config.get("image_count", len(images)), "ONNX image_count"),
        total_detection_count=sum(len(item.detections) for item in images.values()),
        failed_image_count=as_int(summary.get("accuracy_comparison", {}).get("failed_images", 0), "ONNX failed_images"),
        validation_mismatch_count=as_int(validation.get("cuda_internal_mismatches", 0), "ONNX cuda_internal_mismatches"),
        artifact_path=artifact,
        artifact_size_bytes=artifact.stat().st_size if artifact.exists() else None,
    )


def load_tensorrt_results(path: Path, key: str, backend: str, precision: str) -> BackendResult:
    require_dir(path, f"{backend} result directory")
    summary = load_json(path / "summary.json")
    detection_data = load_json(path / "detections.json")
    per_image_rows = read_csv_rows(path / "per_image.csv")

    images: dict[str, ImageResult] = {}
    per_image_timing: dict[str, dict[str, float]] = {}
    for row in per_image_rows:
        image_name = row.get("image_name", "")
        if not image_name:
            raise ValueError(f"{backend} per_image.csv has a row without image_name.")
        per_image_timing.setdefault(
            image_name,
            {
                "inference_ms": as_float(row.get("gpu_execution_mean_ms"), f"{image_name}.gpu_execution_mean_ms"),
                "end_to_end_ms": as_float(row.get("end_to_end_mean_ms"), f"{image_name}.end_to_end_mean_ms"),
            },
        )

    for item in detection_data.get("images", []):
        image_name = Path(str(item.get("image", ""))).name
        if not image_name:
            raise ValueError(f"{backend} detections.json contains an image without a name.")
        if image_name in images:
            raise ValueError(f"Duplicate image_name in {backend} detections.json: {image_name}")
        detections = [
            parse_detection(det, f"{backend} {image_name} detection {index}")
            for index, det in enumerate(item.get("detections", []))
        ]
        timing = per_image_timing.get(image_name, {})
        images[image_name] = ImageResult(
            image_name=image_name,
            detections=detections,
            inference_ms=timing.get("inference_ms"),
            end_to_end_ms=timing.get("end_to_end_ms"),
            status=str(item.get("status", "PASS")),
        )

    timing = summary.get("timing", {})
    artifact = Path(summary.get("engine_path", ""))
    return BackendResult(
        key=key,
        backend=backend,
        precision=precision,
        path=path,
        summary=summary,
        images=images,
        inference_mean_ms=timing.get("gpu_execution", {}).get("mean"),
        inference_median_ms=timing.get("gpu_execution", {}).get("median"),
        inference_p95_ms=timing.get("gpu_execution", {}).get("p95"),
        end_to_end_mean_ms=timing.get("end_to_end", {}).get("mean"),
        end_to_end_median_ms=timing.get("end_to_end", {}).get("median"),
        end_to_end_p95_ms=timing.get("end_to_end", {}).get("p95"),
        image_count=as_int(summary.get("image_count", len(images)), f"{backend}.image_count"),
        total_detection_count=as_int(summary.get("total_detection_count", sum(len(item.detections) for item in images.values())), f"{backend}.total_detection_count"),
        failed_image_count=as_int(summary.get("failed_image_count", 0), f"{backend}.failed_image_count"),
        validation_mismatch_count=as_int(summary.get("validation_mismatch_count", 0), f"{backend}.validation_mismatch_count"),
        artifact_path=artifact,
        artifact_size_bytes=artifact.stat().st_size if artifact.exists() else None,
    )


def validate_image_sets(backends: list[BackendResult]) -> list[str]:
    reference = set(backends[0].images)
    warnings: list[str] = []
    for backend in backends[1:]:
        current = set(backend.images)
        missing = sorted(reference - current)
        extra = sorted(current - reference)
        if missing or extra:
            raise ValueError(
                f"Image set mismatch for {backend.backend}: missing={missing}, extra={extra}"
            )
    for backend in backends:
        if backend.image_count != len(backend.images):
            warnings.append(
                f"{backend.backend} summary image_count={backend.image_count}, parsed images={len(backend.images)}."
            )
    return warnings


def latency_reduction(reference_ms: float | None, candidate_ms: float | None) -> float | None:
    if reference_ms is None or candidate_ms is None or reference_ms <= 0:
        return None
    return (reference_ms - candidate_ms) / reference_ms * 100.0


def speedup(reference_ms: float | None, candidate_ms: float | None) -> float | None:
    if reference_ms is None or candidate_ms is None or candidate_ms <= 0:
        return None
    return reference_ms / candidate_ms


def compare_pair(
    reference: BackendResult,
    candidate: BackendResult,
    match_iou: float,
    strict_confidence_tolerance: float,
    practical_confidence_tolerance: float,
    bbox_tolerance: float,
    confidence_threshold: float,
    threshold_boundary_margin: float,
) -> tuple[list[PairComparison], list[dict[str, Any]]]:
    image_comparisons: list[PairComparison] = []
    detection_rows: list[dict[str, Any]] = []
    for image_name in sorted(reference.images):
        comparison, rows = compare_image(
            image_name=image_name,
            reference_backend=reference.backend,
            candidate_backend=candidate.backend,
            reference=reference.images[image_name].detections,
            candidate=candidate.images[image_name].detections,
            match_iou=match_iou,
            strict_confidence_tolerance=strict_confidence_tolerance,
            practical_confidence_tolerance=practical_confidence_tolerance,
            bbox_tolerance=bbox_tolerance,
            confidence_threshold=confidence_threshold,
            threshold_boundary_margin=threshold_boundary_margin,
        )
        image_comparisons.append(comparison)
        detection_rows.extend(rows)
    return image_comparisons, detection_rows


def aggregate_pair(comparisons: list[PairComparison], reference_backend: str, candidate_backend: str) -> dict[str, Any]:
    statuses = [item.status for item in comparisons]
    overall = "FAIL" if "FAIL" in statuses else ("NUMERICAL_WARNING" if "NUMERICAL_WARNING" in statuses else "PASS")
    threshold_boundary_warning_image_count = sum(
        1
        for item in comparisons
        if item.threshold_boundary_unmatched_reference_count
        or item.threshold_boundary_unmatched_candidate_count
    )
    structural_failure_count = sum(
        1
        for item in comparisons
        if item.structural_unmatched_reference_count
        or item.structural_unmatched_candidate_count
        or item.class_mismatch_count
    )
    return {
        "reference_backend": reference_backend,
        "candidate_backend": candidate_backend,
        "status": overall,
        "image_count": len(comparisons),
        "pass_count": statuses.count("PASS"),
        "numerical_warning_count": statuses.count("NUMERICAL_WARNING"),
        "fail_count": statuses.count("FAIL"),
        "pass_image_count": statuses.count("PASS"),
        "numerical_warning_image_count": statuses.count("NUMERICAL_WARNING"),
        "threshold_boundary_warning_image_count": threshold_boundary_warning_image_count,
        "fail_image_count": statuses.count("FAIL"),
        "matched_detection_count": sum(item.matched_count for item in comparisons),
        "threshold_boundary_unmatched_reference_count": sum(
            item.threshold_boundary_unmatched_reference_count for item in comparisons
        ),
        "threshold_boundary_unmatched_candidate_count": sum(
            item.threshold_boundary_unmatched_candidate_count for item in comparisons
        ),
        "structural_unmatched_reference_count": sum(
            item.structural_unmatched_reference_count for item in comparisons
        ),
        "structural_unmatched_candidate_count": sum(
            item.structural_unmatched_candidate_count for item in comparisons
        ),
        "class_mismatch_count": sum(item.class_mismatch_count for item in comparisons),
        "max_confidence_diff": max((item.confidence_diff_max for item in comparisons), default=0.0),
        "max_bbox_diff": max((item.bbox_diff_max for item in comparisons), default=0.0),
        "min_iou": min((item.bbox_iou_min for item in comparisons), default=1.0),
        "structural_mismatch_count": structural_failure_count,
    }


def backend_summary_row(result: BackendResult) -> dict[str, Any]:
    size_mb = None if result.artifact_size_bytes is None else result.artifact_size_bytes / (1024.0 * 1024.0)
    return {
        "backend": result.backend,
        "precision": result.precision,
        "image_count": result.image_count,
        "total_detection_count": result.total_detection_count,
        "failed_image_count": result.failed_image_count,
        "validation_mismatch_count": result.validation_mismatch_count,
        "inference_mean_ms": result.inference_mean_ms,
        "inference_median_ms": result.inference_median_ms,
        "inference_p95_ms": result.inference_p95_ms,
        "end_to_end_mean_ms": result.end_to_end_mean_ms,
        "end_to_end_median_ms": result.end_to_end_median_ms,
        "end_to_end_p95_ms": result.end_to_end_p95_ms,
        "qps_inference": qps(result.inference_mean_ms),
        "qps_end_to_end": qps(result.end_to_end_mean_ms),
        "model_or_engine_size_bytes": result.artifact_size_bytes,
        "model_or_engine_size_mb": size_mb,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def choose_recommended_backend(backends: list[BackendResult], pair_summaries: list[dict[str, Any]]) -> tuple[str | None, str, str]:
    if any(pair["status"] == "FAIL" for pair in pair_summaries):
        return None, "withheld_structural_failures", "Recommendation withheld because at least one detection comparison failed."
    eligible = [
        backend
        for backend in backends
        if backend.failed_image_count == 0 and backend.validation_mismatch_count == 0 and backend.end_to_end_mean_ms is not None
    ]
    if not eligible:
        return None, "withheld_reliability", "Recommendation withheld because no backend satisfied reliability requirements."
    best = min(eligible, key=lambda item: item.end_to_end_mean_ms or float("inf"))
    has_warning = any(pair["status"] == "NUMERICAL_WARNING" for pair in pair_summaries)
    if has_warning:
        return (
            best.backend,
            "recommended_with_numerical_warnings",
            (
                f"{best.backend} had the lowest mean end-to-end latency, with no structural detection failures. "
                "Remaining differences are limited to confidence-threshold boundary detections or practical numerical confidence tolerance."
            ),
        )
    return best.backend, "recommended", f"{best.backend} passed validation and had the lowest mean end-to-end latency."


def write_report(
    path: Path,
    backend_rows: list[dict[str, Any]],
    pair_summaries: list[dict[str, Any]],
    performance_pairs: list[dict[str, Any]],
    recommended_backend: str | None,
    recommendation_status: str,
    recommendation_reason: str,
    warnings: list[str],
    confidence_threshold: float,
    threshold_boundary_margin: float,
) -> None:
    def fmt(value: Any) -> str:
        if value is None or value == "":
            return "N/A"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "# C++ Inference Backend Final Comparison Report",
        "",
        "## 1. Purpose",
        "Compare ONNX Runtime CUDA, Native TensorRT FP32, and Native TensorRT FP16 using existing benchmark result files only.",
        "",
        "## 2. Compared Backends",
        "",
        "| Backend | Precision | Images | Detections | Inference mean ms | End-to-end mean ms | Failed images | Validation mismatches |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in backend_rows:
        lines.append(
            "| {backend} | {precision} | {image_count} | {total_detection_count} | {inference_mean_ms} | {end_to_end_mean_ms} | {failed_image_count} | {validation_mismatch_count} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## 3. Performance Speedups",
            "",
            "| Comparison | Inference speedup | Inference reduction | End-to-end speedup | End-to-end reduction |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in performance_pairs:
        lines.append(
            f"| {row['comparison']} | {fmt(row['inference_speedup'])}x | {fmt(row['inference_latency_reduction_percent'])}% | "
            f"{fmt(row['end_to_end_speedup'])}x | {fmt(row['end_to_end_latency_reduction_percent'])}% |"
        )
    lines.extend(
        [
            "",
            "## 4. Detection Validation",
            "",
            f"Confidence threshold boundary margin: threshold={fmt(confidence_threshold)}, margin={fmt(threshold_boundary_margin)}.",
            "",
            "| Comparison | Status | PASS | Warning | Boundary warning | FAIL | Structural failures | Max confidence diff | Max bbox diff | Min IoU |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pair_summaries:
        lines.append(
            f"| {row['reference_backend']} vs {row['candidate_backend']} | {row['status']} | {row['pass_count']} | "
            f"{row['numerical_warning_count']} | {row['threshold_boundary_warning_image_count']} | {row['fail_count']} | "
            f"{row['structural_mismatch_count']} | {fmt(row['max_confidence_diff'])} | "
            f"{fmt(row['max_bbox_diff'])} | {fmt(row['min_iou'])} |"
        )
    lines.extend(
        [
            "",
            "## 5. Recommendation",
            "",
            f"Recommended backend: {recommended_backend or 'N/A'}",
            f"Recommendation status: {recommendation_status}",
            "",
            recommendation_reason,
            "",
            "Threshold-boundary unmatched detections are treated as `NUMERICAL_WARNING` only when every unmatched detection is within the configured confidence boundary. Class mismatches, bbox mismatches, and unmatched detections outside that boundary remain `FAIL`.",
            "",
            "## 6. Limitations",
            "",
            "- This report uses previously generated benchmark files and does not rerun inference.",
            "- ONNX Runtime CUDA end-to-end p95 is computed from timing_runs.csv because summary.json does not store it directly.",
        ]
    )
    if warnings:
        lines.extend(["", "## 7. Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_comparison(args: argparse.Namespace) -> dict[str, Any]:
    onnx = load_onnx_results(Path(args.onnx))
    trt_fp32 = load_tensorrt_results(Path(args.tensorrt_fp32), "tensorrt_fp32", "Native TensorRT FP32", "FP32")
    trt_fp16 = load_tensorrt_results(Path(args.tensorrt_fp16), "tensorrt_fp16", "Native TensorRT FP16", "FP16")
    backends = [onnx, trt_fp32, trt_fp16]
    warnings = validate_image_sets(backends)

    pair_specs = [
        (onnx, trt_fp32),
        (onnx, trt_fp16),
        (trt_fp32, trt_fp16),
    ]
    all_image_comparisons: dict[tuple[str, str], list[PairComparison]] = {}
    detection_rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    for reference, candidate in pair_specs:
        practical_confidence_tolerance = (
            args.fp16_practical_confidence_tolerance
            if reference.key == "tensorrt_fp16" or candidate.key == "tensorrt_fp16"
            else args.practical_confidence_tolerance
        )
        image_comparisons, rows = compare_pair(
            reference,
            candidate,
            args.match_iou,
            args.strict_confidence_tolerance,
            practical_confidence_tolerance,
            args.bbox_tolerance,
            args.confidence_threshold,
            args.threshold_boundary_margin,
        )
        all_image_comparisons[(reference.key, candidate.key)] = image_comparisons
        detection_rows.extend(rows)
        pair_summaries.append(aggregate_pair(image_comparisons, reference.backend, candidate.backend))

    backend_rows = [backend_summary_row(item) for item in backends]
    by_key = {backend.key: backend for backend in backends}
    performance_pairs = []
    for left_key, right_key, label in [
        ("onnx_cuda", "tensorrt_fp32", "ORT CUDA vs TensorRT FP32"),
        ("onnx_cuda", "tensorrt_fp16", "ORT CUDA vs TensorRT FP16"),
        ("tensorrt_fp32", "tensorrt_fp16", "TensorRT FP32 vs FP16"),
    ]:
        left = by_key[left_key]
        right = by_key[right_key]
        performance_pairs.append(
            {
                "comparison": label,
                "inference_speedup": speedup(left.inference_mean_ms, right.inference_mean_ms),
                "inference_latency_reduction_percent": latency_reduction(left.inference_mean_ms, right.inference_mean_ms),
                "end_to_end_speedup": speedup(left.end_to_end_mean_ms, right.end_to_end_mean_ms),
                "end_to_end_latency_reduction_percent": latency_reduction(left.end_to_end_mean_ms, right.end_to_end_mean_ms),
            }
        )

    recommended_backend, recommendation_status, recommendation_reason = choose_recommended_backend(backends, pair_summaries)
    overall_status = "FAIL" if any(item["status"] == "FAIL" for item in pair_summaries) else (
        "NUMERICAL_WARNING" if any(item["status"] == "NUMERICAL_WARNING" for item in pair_summaries) else "PASS"
    )

    per_image_rows: list[dict[str, Any]] = []
    fp32_map = {item.image_name: item for item in all_image_comparisons[("onnx_cuda", "tensorrt_fp32")]}
    fp16_map = {item.image_name: item for item in all_image_comparisons[("onnx_cuda", "tensorrt_fp16")]}
    fp32_fp16_map = {item.image_name: item for item in all_image_comparisons[("tensorrt_fp32", "tensorrt_fp16")]}
    for image_name in sorted(onnx.images):
        per_image_rows.append(
            {
                "image_name": image_name,
                "onnx_cuda_detection_count": len(onnx.images[image_name].detections),
                "tensorrt_fp32_detection_count": len(trt_fp32.images[image_name].detections),
                "tensorrt_fp16_detection_count": len(trt_fp16.images[image_name].detections),
                "onnx_cuda_inference_ms": onnx.images[image_name].inference_ms,
                "tensorrt_fp32_inference_ms": trt_fp32.images[image_name].inference_ms,
                "tensorrt_fp16_inference_ms": trt_fp16.images[image_name].inference_ms,
                "onnx_cuda_end_to_end_ms": onnx.images[image_name].end_to_end_ms,
                "tensorrt_fp32_end_to_end_ms": trt_fp32.images[image_name].end_to_end_ms,
                "tensorrt_fp16_end_to_end_ms": trt_fp16.images[image_name].end_to_end_ms,
                "fp32_status": fp32_map[image_name].status,
                "fp16_status": fp16_map[image_name].status,
                "fp32_vs_fp16_status": fp32_fp16_map[image_name].status,
            }
        )

    failure_rows = [
        {
            "image_name": item.image_name,
            "comparison": f"{item.reference_backend} vs {item.candidate_backend}",
            "status": item.status,
            "reason": item.reason,
            "max_confidence_diff": item.confidence_diff_max,
            "max_bbox_diff": item.bbox_diff_max,
            "min_iou": item.bbox_iou_min,
        }
        for comparisons in all_image_comparisons.values()
        for item in comparisons
        if item.status != "PASS"
    ]
    threshold_boundary_warning_count = sum(
        item.threshold_boundary_unmatched_reference_count + item.threshold_boundary_unmatched_candidate_count
        for comparisons in all_image_comparisons.values()
        for item in comparisons
    )
    structural_failure_count = sum(
        1
        for comparisons in all_image_comparisons.values()
        for item in comparisons
        if item.status == "FAIL"
        and (
            item.structural_unmatched_reference_count
            or item.structural_unmatched_candidate_count
            or item.class_mismatch_count
        )
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_backend": onnx.backend,
        "confidence_threshold": args.confidence_threshold,
        "threshold_boundary_margin": args.threshold_boundary_margin,
        "fp16_practical_confidence_tolerance": args.fp16_practical_confidence_tolerance,
        "threshold_boundary_warning_count": threshold_boundary_warning_count,
        "structural_failure_count": structural_failure_count,
        "comparison_settings": {
            "match_iou": args.match_iou,
            "strict_confidence_tolerance": args.strict_confidence_tolerance,
            "practical_confidence_tolerance": args.practical_confidence_tolerance,
            "fp16_practical_confidence_tolerance": args.fp16_practical_confidence_tolerance,
            "confidence_threshold": args.confidence_threshold,
            "threshold_boundary_margin": args.threshold_boundary_margin,
            "bbox_tolerance": args.bbox_tolerance,
        },
        "backend_summaries": backend_rows,
        "pairwise_performance": performance_pairs,
        "pairwise_detection_validation": pair_summaries,
        "overall_status": overall_status,
        "recommended_backend": recommended_backend,
        "recommendation_status": recommendation_status,
        "recommendation_reason": recommendation_reason,
        "warnings": warnings,
        "_per_image_rows": per_image_rows,
        "_detection_rows": detection_rows,
        "_failure_rows": failure_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare existing C++ ONNX CUDA and Native TensorRT benchmark results.")
    parser.add_argument("--onnx", required=True, help="ONNX Runtime CUDA benchmark result directory.")
    parser.add_argument("--tensorrt-fp32", required=True, help="Native TensorRT FP32 benchmark result directory.")
    parser.add_argument("--tensorrt-fp16", required=True, help="Native TensorRT FP16 benchmark result directory.")
    parser.add_argument("--output", required=True, help="Output directory for comparison report files.")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--strict-confidence-tolerance", type=float, default=0.001)
    parser.add_argument("--practical-confidence-tolerance", type=float, default=0.002)
    parser.add_argument("--fp16-practical-confidence-tolerance", type=float, default=0.005)
    parser.add_argument("--confidence-threshold", type=float, default=0.15)
    parser.add_argument("--threshold-boundary-margin", type=float, default=0.001)
    parser.add_argument("--bbox-tolerance", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    failure_dir = output / "failure_cases"
    failure_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison(args)
    per_image_rows = comparison.pop("_per_image_rows")
    detection_rows = comparison.pop("_detection_rows")
    failure_rows = comparison.pop("_failure_rows")

    (output / "summary.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(
        output / "backend_comparison.csv",
        comparison["backend_summaries"],
        [
            "backend",
            "precision",
            "image_count",
            "total_detection_count",
            "failed_image_count",
            "validation_mismatch_count",
            "inference_mean_ms",
            "inference_median_ms",
            "inference_p95_ms",
            "end_to_end_mean_ms",
            "end_to_end_median_ms",
            "end_to_end_p95_ms",
            "qps_inference",
            "qps_end_to_end",
            "model_or_engine_size_mb",
        ],
    )
    write_csv(
        output / "per_image_comparison.csv",
        per_image_rows,
        [
            "image_name",
            "onnx_cuda_detection_count",
            "tensorrt_fp32_detection_count",
            "tensorrt_fp16_detection_count",
            "onnx_cuda_inference_ms",
            "tensorrt_fp32_inference_ms",
            "tensorrt_fp16_inference_ms",
            "onnx_cuda_end_to_end_ms",
            "tensorrt_fp32_end_to_end_ms",
            "tensorrt_fp16_end_to_end_ms",
            "fp32_status",
            "fp16_status",
            "fp32_vs_fp16_status",
        ],
    )
    write_csv(
        output / "detection_comparison.csv",
        detection_rows,
        [
            "image_name",
            "reference_backend",
            "candidate_backend",
            "reference_index",
            "candidate_index",
            "class_id",
            "class_name",
            "reference_confidence",
            "candidate_confidence",
            "confidence_diff",
            "reference_x1",
            "reference_y1",
            "reference_x2",
            "reference_y2",
            "candidate_x1",
            "candidate_y1",
            "candidate_x2",
            "candidate_y2",
            "bbox_max_abs_diff",
            "iou",
            "confidence_threshold",
            "threshold_boundary_margin",
            "boundary_distance",
            "is_threshold_boundary",
            "status",
        ],
    )
    write_csv(
        failure_dir / "manifest.csv",
        failure_rows,
        ["image_name", "comparison", "status", "reason", "max_confidence_diff", "max_bbox_diff", "min_iou"],
    )
    write_report(
        output / "report.md",
        comparison["backend_summaries"],
        comparison["pairwise_detection_validation"],
        comparison["pairwise_performance"],
        comparison["recommended_backend"],
        comparison["recommendation_status"],
        comparison["recommendation_reason"],
        comparison["warnings"],
        comparison["confidence_threshold"],
        comparison["threshold_boundary_margin"],
    )

    print(f"Wrote backend comparison report to: {output}")
    print(f"Overall status: {comparison['overall_status']}")
    print(f"Recommended backend: {comparison['recommended_backend'] or 'N/A'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
