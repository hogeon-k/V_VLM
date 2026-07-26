from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.compare_cpp_inference_backends import (
    BackendResult,
    Detection,
    bbox_iou,
    build_comparison,
    compare_image,
    choose_recommended_backend,
    latency_reduction,
    main,
    parse_args,
    qps,
    speedup,
)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(root: Path) -> tuple[Path, Path, Path]:
    onnx = root / "onnx"
    fp32 = root / "fp32"
    fp16 = root / "fp16"
    images = ["a.jpg", "b.jpg"]

    write_json(
        onnx / "summary.json",
        {
            "config": {"model": str(root / "best.onnx"), "image_count": 2},
            "accuracy_comparison": {"failed_images": 0},
            "timing": {
                "cuda": {
                    "session_mean": 8.0,
                    "session_median": 7.5,
                    "session_p95": 9.0,
                    "total_mean": 18.0,
                    "total_median": 17.0,
                }
            },
            "validation": {"cuda_internal_mismatches": 0},
        },
    )
    write_csv(
        onnx / "image_results.csv",
        [
            "image",
            "cuda_detection_count",
            "cuda_session_mean_ms",
            "cuda_total_mean_ms",
            "status",
        ],
        [
            {"image": "a.jpg", "cuda_detection_count": 1, "cuda_session_mean_ms": 8.0, "cuda_total_mean_ms": 18.0, "status": "PASS"},
            {"image": "b.jpg", "cuda_detection_count": 1, "cuda_session_mean_ms": 8.2, "cuda_total_mean_ms": 18.2, "status": "PASS"},
        ],
    )
    write_csv(
        onnx / "timing_runs.csv",
        ["image", "provider", "repeat_index", "session_run_ms", "total_ms"],
        [
            {"image": "a.jpg", "provider": "cuda", "repeat_index": 0, "session_run_ms": 8.0, "total_ms": 18.0},
            {"image": "a.jpg", "provider": "cuda", "repeat_index": 1, "session_run_ms": 9.0, "total_ms": 19.0},
        ],
    )
    for image in images:
        write_json(
            onnx / "cuda" / "predictions" / f"{Path(image).stem}.json",
            {
                "provider": "CUDAExecutionProvider",
                "detections": [
                    {"class_id": 1, "class_name": "short", "confidence": 0.9, "bbox": [0.0, 0.0, 10.0, 10.0]}
                ],
            },
        )

    def write_trt(path: Path, label: str, confidence: float, inference: float, total: float) -> None:
        write_json(
            path / "summary.json",
            {
                "engine_path": str(root / f"{label}.engine"),
                "image_count": 2,
                "total_detection_count": 2,
                "failed_image_count": 0,
                "validation_mismatch_count": 0,
                "timing": {
                    "gpu_execution": {"mean": inference, "median": inference, "p95": inference + 0.1},
                    "end_to_end": {"mean": total, "median": total, "p95": total + 0.1},
                },
            },
        )
        write_csv(
            path / "per_image.csv",
            [
                "image_name",
                "engine_label",
                "detection_count",
                "detection_index",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "gpu_execution_mean_ms",
                "end_to_end_mean_ms",
            ],
            [
                {
                    "image_name": image,
                    "engine_label": label,
                    "detection_count": 1,
                    "detection_index": 0,
                    "class_id": 1,
                    "class_name": "short",
                    "confidence": confidence,
                    "x1": 0.0,
                    "y1": 0.0,
                    "x2": 10.0,
                    "y2": 10.0,
                    "gpu_execution_mean_ms": inference,
                    "end_to_end_mean_ms": total,
                }
                for image in images
            ],
        )
        write_json(
            path / "detections.json",
            {
                "images": [
                    {
                        "image": image,
                        "status": "PASS",
                        "detections": [
                            {
                                "class_id": 1,
                                "class_name": "short",
                                "confidence": confidence,
                                "bbox": [0.0, 0.0, 10.0, 10.0],
                            }
                        ],
                    }
                    for image in images
                ]
            },
        )

    write_trt(fp32, "fp32", 0.9005, 3.0, 15.0)
    write_trt(fp16, "fp16", 0.9015, 2.0, 14.0)
    (root / "best.onnx").write_bytes(b"onnx")
    (root / "fp32.engine").write_bytes(b"fp32")
    (root / "fp16.engine").write_bytes(b"fp16")
    return onnx, fp32, fp16


def test_iou_and_matching_pass_warning_fail() -> None:
    ref = [Detection(1, "short", 0.9, (0.0, 0.0, 10.0, 10.0))]
    same = [Detection(1, "short", 0.9005, (0.0, 0.0, 10.0, 10.0))]
    warning = [Detection(1, "short", 0.9015, (0.0, 0.0, 10.0, 10.0))]
    fail_conf = [Detection(1, "short", 0.91, (0.0, 0.0, 10.0, 10.0))]
    fail_bbox = [Detection(1, "short", 0.9, (2.0, 0.0, 12.0, 10.0))]
    fail_class = [Detection(2, "missing_hole", 0.9, (0.0, 0.0, 10.0, 10.0))]

    assert bbox_iou(ref[0], same[0]) == 1.0
    assert compare_image("x.jpg", "ref", "same", ref, same, 0.5, 0.001, 0.002, 1.0, 0.15, 0.001)[0].status == "PASS"
    assert compare_image("x.jpg", "ref", "warn", ref, warning, 0.5, 0.001, 0.002, 1.0, 0.15, 0.001)[0].status == "NUMERICAL_WARNING"
    assert compare_image("x.jpg", "ref", "fail", ref, fail_conf, 0.5, 0.001, 0.002, 1.0, 0.15, 0.001)[0].status == "FAIL"
    assert compare_image("x.jpg", "ref", "fail", ref, fail_bbox, 0.5, 0.001, 0.002, 1.0, 0.15, 0.001)[0].status == "FAIL"
    assert compare_image("x.jpg", "ref", "fail", ref, fail_class, 0.5, 0.001, 0.002, 1.0, 0.15, 0.001)[0].status == "FAIL"
    assert compare_image("x.jpg", "ref", "fail", ref, [], 0.5, 0.001, 0.002, 1.0, 0.15, 0.001)[0].status == "FAIL"


def compare_status(
    reference: list[Detection],
    candidate: list[Detection],
    practical_confidence_tolerance: float = 0.002,
) -> tuple[str, list[dict[str, object]]]:
    comparison, rows = compare_image(
        "x.jpg",
        "ref",
        "cand",
        reference,
        candidate,
        0.5,
        0.001,
        practical_confidence_tolerance,
        1.0,
        0.15,
        0.001,
    )
    return comparison.status, rows


def test_unmatched_candidate_near_confidence_threshold_is_warning() -> None:
    status, rows = compare_status([], [Detection(1, "short", 0.1505, (0.0, 0.0, 10.0, 10.0))])

    assert status == "NUMERICAL_WARNING"
    assert rows[0]["status"] == "WARNING_UNMATCHED_CANDIDATE_THRESHOLD_BOUNDARY"


def test_unmatched_reference_near_confidence_threshold_is_warning() -> None:
    status, rows = compare_status([Detection(1, "short", 0.1495, (0.0, 0.0, 10.0, 10.0))], [])

    assert status == "NUMERICAL_WARNING"
    assert rows[0]["status"] == "WARNING_UNMATCHED_REFERENCE_THRESHOLD_BOUNDARY"


def test_unmatched_detection_outside_confidence_threshold_boundary_fails() -> None:
    status, rows = compare_status([], [Detection(1, "short", 0.152, (0.0, 0.0, 10.0, 10.0))])

    assert status == "FAIL"
    assert rows[0]["status"] == "FAIL_UNMATCHED_CANDIDATE"


def test_multiple_unmatched_all_boundary_are_warning() -> None:
    status, _rows = compare_status(
        [Detection(1, "short", 0.1495, (0.0, 0.0, 10.0, 10.0))],
        [Detection(1, "short", 0.1505, (20.0, 20.0, 30.0, 30.0))],
    )

    assert status == "NUMERICAL_WARNING"


def test_any_unmatched_outside_boundary_fails() -> None:
    status, _rows = compare_status(
        [Detection(1, "short", 0.1495, (0.0, 0.0, 10.0, 10.0))],
        [Detection(1, "short", 0.152, (20.0, 20.0, 30.0, 30.0))],
    )

    assert status == "FAIL"


def test_class_mismatch_with_boundary_unmatched_still_fails() -> None:
    status, _rows = compare_status(
        [
            Detection(1, "short", 0.9, (0.0, 0.0, 10.0, 10.0)),
            Detection(1, "short", 0.1495, (20.0, 20.0, 30.0, 30.0)),
        ],
        [
            Detection(2, "missing_hole", 0.9, (0.0, 0.0, 10.0, 10.0)),
        ],
    )

    assert status == "FAIL"


def test_fp32_and_fp16_practical_confidence_tolerances() -> None:
    ref = [Detection(1, "short", 0.9, (0.0, 0.0, 10.0, 10.0))]

    assert compare_status(ref, [Detection(1, "short", 0.9015, (0.0, 0.0, 10.0, 10.0))], 0.002)[0] == "NUMERICAL_WARNING"
    assert compare_status(ref, [Detection(1, "short", 0.9046, (0.0, 0.0, 10.0, 10.0))], 0.005)[0] == "NUMERICAL_WARNING"
    assert compare_status(ref, [Detection(1, "short", 0.9051, (0.0, 0.0, 10.0, 10.0))], 0.005)[0] == "FAIL"


def test_speedup_qps_and_report_generation(tmp_path: Path, monkeypatch) -> None:
    onnx, fp32, fp16 = make_fixture(tmp_path)
    output = tmp_path / "out"
    args = argparse.Namespace(
        onnx=str(onnx),
        tensorrt_fp32=str(fp32),
        tensorrt_fp16=str(fp16),
        output=str(output),
        match_iou=0.5,
        strict_confidence_tolerance=0.001,
        practical_confidence_tolerance=0.002,
        fp16_practical_confidence_tolerance=0.005,
        confidence_threshold=0.15,
        threshold_boundary_margin=0.001,
        bbox_tolerance=1.0,
    )
    comparison = build_comparison(args)

    assert qps(10.0) == 100.0
    assert speedup(8.0, 2.0) == 4.0
    assert latency_reduction(8.0, 2.0) == 75.0
    assert comparison["overall_status"] == "NUMERICAL_WARNING"
    assert comparison["recommended_backend"] == "Native TensorRT FP16"
    assert comparison["recommendation_status"] == "recommended_with_numerical_warnings"

    monkeypatch.setattr(
        "sys.argv",
        [
            "compare",
            "--onnx",
            str(onnx),
            "--tensorrt-fp32",
            str(fp32),
            "--tensorrt-fp16",
            str(fp16),
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    for name in [
        "summary.json",
        "backend_comparison.csv",
        "per_image_comparison.csv",
        "detection_comparison.csv",
        "report.md",
        "failure_cases/manifest.csv",
    ]:
        assert (output / name).exists()


def test_threshold_warning_only_keeps_recommendation_available(tmp_path: Path) -> None:
    backend = BackendResult(
        key="fast",
        backend="Native TensorRT FP16",
        precision="FP16",
        path=tmp_path,
        summary={},
        images={},
        end_to_end_mean_ms=1.0,
    )
    recommended, status, reason = choose_recommended_backend(
        [backend],
        [{"status": "NUMERICAL_WARNING", "structural_mismatch_count": 0}],
    )

    assert recommended == "Native TensorRT FP16"
    assert status == "recommended_with_numerical_warnings"
    assert "no structural detection failures" in reason


def test_structural_fail_withholds_recommendation(tmp_path: Path) -> None:
    backend = BackendResult(
        key="fast",
        backend="Native TensorRT FP16",
        precision="FP16",
        path=tmp_path,
        summary={},
        images={},
        end_to_end_mean_ms=1.0,
    )
    recommended, status, _reason = choose_recommended_backend([backend], [{"status": "FAIL"}])

    assert recommended is None
    assert status == "withheld_structural_failures"


def test_cli_parser_has_threshold_and_fp16_options(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare",
            "--onnx",
            "onnx",
            "--tensorrt-fp32",
            "fp32",
            "--tensorrt-fp16",
            "fp16",
            "--output",
            "out",
            "--confidence-threshold",
            "0.2",
            "--threshold-boundary-margin",
            "0.003",
            "--fp16-practical-confidence-tolerance",
            "0.006",
        ],
    )
    args = parse_args()

    assert args.confidence_threshold == 0.2
    assert args.threshold_boundary_margin == 0.003
    assert args.fp16_practical_confidence_tolerance == 0.006
