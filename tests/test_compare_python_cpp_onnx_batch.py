from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts.compare_python_cpp_onnx_batch import (
    TimedJsonResult,
    TimingSample,
    bbox_iou,
    build_summary,
    collect_images,
    compare_results,
    exit_code_for_status,
    extensions_from_arg,
    final_status,
    judge_status,
    match_detections,
    run_batch,
    timing_stats,
)


def make_args(tmp_path: Path, image_dir: Path) -> argparse.Namespace:
    model = tmp_path / "best.onnx"
    cpp_exe = tmp_path / "pcb_onnx_infer.exe"
    model.write_bytes(b"onnx")
    cpp_exe.write_bytes(b"exe")
    return argparse.Namespace(
        images=image_dir,
        model=model,
        metadata=tmp_path / "model_metadata.json",
        cpp_exe=cpp_exe,
        output=tmp_path / "batch",
        imgsz=960,
        conf=0.15,
        iou=0.7,
        match_iou=0.5,
        warmup=0,
        repeat=2,
        extensions=".jpg,.jpeg,.png,.bmp",
    )


def write_images(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"fake")


def det(class_id: int = 1, confidence: float = 0.9, bbox: list[float] | None = None) -> dict[str, object]:
    return {
        "class_id": class_id,
        "class_name": "short",
        "confidence": confidence,
        "bbox": bbox or [0.0, 0.0, 10.0, 10.0],
    }


def result(image_name: str, detections: list[dict[str, object]], backend: str = "python", total: float = 10.0) -> TimedJsonResult:
    data = {
        "image": image_name,
        "provider": "CPUExecutionProvider",
        "timing_ms": {"preprocess": 1.0, "inference": total - 2.0, "postprocess": 1.0, "total": total},
        "detections": detections,
    }
    return TimedJsonResult(
        data=data,
        timings=[
            TimingSample(image_name, backend, 0, 1.0, total - 2.0, 1.0, total, "CPUExecutionProvider"),
            TimingSample(image_name, backend, 1, 1.0, total, 1.0, total + 2.0, "CPUExecutionProvider"),
        ],
    )


def test_collect_images_sorts_and_filters_supported_extensions(tmp_path: Path) -> None:
    write_images(tmp_path, ["b.png", "a.jpg", "c.txt", "d.BMP", "e.jpeg"])

    images = collect_images(tmp_path, extensions=(".jpg", ".jpeg", ".png", ".bmp"))

    assert [image.name for image in images] == ["a.jpg", "b.png", "d.BMP", "e.jpeg"]
    assert extensions_from_arg("jpg,png") == (".jpg", ".png")


def test_bbox_iou_and_global_class_aware_matching() -> None:
    python_detections = [det(1, bbox=[0, 0, 10, 10]), det(1, bbox=[30, 30, 40, 40]), det(2, bbox=[0, 0, 10, 10])]
    cpp_detections = [det(1, bbox=[30, 30, 40, 40]), det(1, bbox=[0, 0, 10, 10]), det(0, bbox=[0, 0, 10, 10])]

    matches = match_detections(python_detections, cpp_detections, match_iou=0.5)

    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert [match["status"] for match in matches].count("MATCHED") == 2
    assert [match["status"] for match in matches].count("PYTHON_ONLY") == 1
    assert [match["status"] for match in matches].count("CPP_ONLY") == 1


def test_compare_results_pass_warning_and_fail_judgement() -> None:
    image = Path("one.jpg")
    passed = compare_results(image, result("one.jpg", [det(1, 0.9)]), result("one.jpg", [det(1, 0.900001)], "cpp"), 0.5)
    warning = compare_results(image, result("one.jpg", [det(1, 0.9)]), result("one.jpg", [det(1, 0.9005, [0, 0, 10, 9.95])], "cpp"), 0.5)
    failed = compare_results(image, result("one.jpg", [det()]), result("one.jpg", [], "cpp"), 0.5)

    assert passed.status == "PASS"
    assert warning.status == "WARNING"
    assert failed.status == "FAIL"
    assert judge_status(1, 1, 1, 0, 0, True, 0.0, 0.98)[0] == "FAIL"


def test_timing_stats_summary_and_exit_codes(tmp_path: Path) -> None:
    image = Path("one.jpg")
    comparisons = [
        compare_results(image, result("one.jpg", [det()], total=20.0), result("one.jpg", [det()], "cpp", total=10.0), 0.5)
    ]
    args = make_args(tmp_path, tmp_path)
    summary = build_summary(args, [image], comparisons, comparisons[0].timing_rows)

    assert timing_stats([1.0, 2.0, 3.0])["p95_ms"] == pytest.approx(2.9)
    assert summary["final_status"] == "PASS"
    assert summary["detection_summary"]["matched_detection_count"] == 1
    assert summary["timing_summary"]["speedup"]["total_mean"] == pytest.approx(21.0 / 11.0)
    assert final_status(comparisons) == "PASS"
    assert exit_code_for_status("PASS") == 0
    assert exit_code_for_status("WARNING") == 1
    assert exit_code_for_status("FAIL") == 2


def test_run_batch_writes_json_csv_and_failure_cases(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    write_images(image_dir, ["pass.jpg", "fail.jpg"])
    args = make_args(tmp_path, image_dir)

    def python_runner(image_path: Path) -> TimedJsonResult:
        return result(image_path.name, [det()], total=20.0)

    def cpp_runner(image_path: Path) -> TimedJsonResult:
        detections = [] if image_path.stem == "fail" else [det()]
        return result(image_path.name, detections, "cpp", total=10.0)

    batch = run_batch(args, python_runner=python_runner, cpp_runner=cpp_runner)

    assert batch.exit_code == 2
    assert (args.output / "summary.json").exists()
    assert (args.output / "python" / "pass.json").exists()
    assert (args.output / "cpp" / "pass.json").exists()
    assert (args.output / "comparisons" / "fail.json").exists()
    assert (args.output / "failure_cases" / "fail.jpg").exists()
    with (args.output / "per_image.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["status"] for row in rows} == {"PASS", "FAIL"}
    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["fail_count"] == 1


def test_empty_folder_returns_code_2_and_writes_outputs(tmp_path: Path) -> None:
    image_dir = tmp_path / "empty"
    image_dir.mkdir()
    args = make_args(tmp_path, image_dir)

    batch = run_batch(args, python_runner=lambda path: result(path.name, []), cpp_runner=lambda path: result(path.name, [], "cpp"))

    assert batch.exit_code == 2
    assert (args.output / "summary.json").exists()
    assert (args.output / "per_image.csv").exists()


def test_cpp_execution_failure_is_reported_and_processing_continues(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    write_images(image_dir, ["bad.jpg", "good.jpg"])
    args = make_args(tmp_path, image_dir)

    def cpp_runner(image_path: Path) -> TimedJsonResult:
        if image_path.stem == "bad":
            raise RuntimeError("cpp failed")
        return result(image_path.name, [], "cpp")

    batch = run_batch(args, python_runner=lambda path: result(path.name, []), cpp_runner=cpp_runner)

    assert batch.summary["final_status"] == "ERROR"
    assert batch.summary["counts"]["error_count"] == 1
    assert batch.summary["counts"]["pass_count"] == 1
    assert (args.output / "failure_cases" / "bad.jpg").exists()
