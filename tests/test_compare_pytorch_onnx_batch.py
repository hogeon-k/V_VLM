from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from model.defect_info import Detection
from scripts.compare_pytorch_onnx_batch import (
    TimedDetectionsResult,
    TimingSample,
    calculate_speedup,
    collect_images,
    exit_code_for_status,
    final_status,
    format_error_text,
    format_mismatch_text,
    judge_status,
    percentile,
    run_batch,
    timing_stats,
    validate_args,
)


def make_args(tmp_path: Path, image_dir: Path) -> argparse.Namespace:
    pt_model = tmp_path / "best.pt"
    onnx_model = tmp_path / "best.onnx"
    pt_model.write_bytes(b"pt")
    onnx_model.write_bytes(b"onnx")
    return argparse.Namespace(
        images=image_dir,
        pytorch_model=pt_model,
        onnx_model=onnx_model,
        output=tmp_path / "report",
        imgsz=960,
        conf=0.15,
        iou=0.7,
        match_iou=0.5,
        device="0",
        warmup=0,
        repeat=2,
        save_all_images=False,
        fail_on_warning=False,
        extensions=".jpg,.png",
        recursive=False,
        max_images=None,
        seed=None,
    )


def write_images(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"fake")


def timed_result(detections: list[Detection], backend: str = "pytorch", total: float = 10.0) -> TimedDetectionsResult:
    provider = "0" if backend == "pytorch" else "CPUExecutionProvider"
    return TimedDetectionsResult(
        detections=detections,
        timings=[
            TimingSample(backend, 0, 1.0, total - 2.0, 1.0, total, provider),
            TimingSample(backend, 1, 1.0, total, 1.0, total + 2.0, provider),
        ],
        providers=[provider] if backend == "onnx" else [],
    )


def det(confidence: float = 0.9, class_id: int = 1, bbox: tuple[int, int, int, int] = (0, 0, 10, 10)) -> Detection:
    return Detection(class_id, "short", confidence, *bbox)


def test_collect_images_sorts_and_filters_extensions(tmp_path: Path) -> None:
    write_images(tmp_path, ["b.png", "a.jpg", "c.txt", "d.JPG"])

    images = collect_images(tmp_path, extensions=(".jpg", ".png"))

    assert [image.name for image in images] == ["a.jpg", "b.png", "d.JPG"]


def test_collect_images_recursive_and_max_images(tmp_path: Path) -> None:
    write_images(tmp_path, ["b.jpg"])
    write_images(tmp_path / "nested", ["a.jpg"])

    images = collect_images(tmp_path, extensions=(".jpg",), recursive=True, max_images=1)

    assert [image.name for image in images] == ["b.jpg"]


def test_collect_images_seed_is_deterministic(tmp_path: Path) -> None:
    write_images(tmp_path, ["a.jpg", "b.jpg", "c.jpg"])

    first = collect_images(tmp_path, extensions=(".jpg",), seed=7)
    second = collect_images(tmp_path, extensions=(".jpg",), seed=7)

    assert [image.name for image in first] == [image.name for image in second]


def test_validate_args_rejects_empty_image_folder(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    args = make_args(tmp_path, image_dir)
    args.repeat = 0
    args.warmup = -1

    errors = validate_args(args)

    assert "--repeat must be >= 1" in errors
    assert "--warmup must be >= 0" in errors


def test_validate_args_rejects_invalid_thresholds_and_size(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    write_images(image_dir, ["a.jpg"])
    args = make_args(tmp_path, image_dir)
    args.imgsz = 0
    args.conf = 1.5
    args.iou = -0.1
    args.match_iou = 2.0

    errors = validate_args(args)

    assert "--imgsz must be > 0" in errors
    assert "--conf must be between 0 and 1" in errors
    assert "--iou must be between 0 and 1" in errors
    assert "--match-iou must be between 0 and 1" in errors


def test_timing_stats_and_p95() -> None:
    stats = timing_stats([1.0, 2.0, 3.0, 4.0])

    assert stats["mean_ms"] == pytest.approx(2.5)
    assert stats["median_ms"] == pytest.approx(2.5)
    assert stats["p95_ms"] == pytest.approx(3.85)
    assert percentile([10.0], 95) == pytest.approx(10.0)


def test_speedup_uses_pytorch_time_divided_by_onnx_time() -> None:
    assert calculate_speedup(20.0, 10.0) == pytest.approx(2.0)
    assert calculate_speedup(20.0, 0.0) is None


def test_pass_warning_fail_error_judgements() -> None:
    assert judge_status(0, 0, 0, 0, 0, True, 0.0, 1.0)[0] == "PASS"
    assert judge_status(1, 1, 1, 0, 0, True, 0.005, 0.995)[0] == "PASS"
    assert judge_status(1, 1, 1, 0, 0, True, 0.02, 0.995)[0] == "WARNING"
    assert judge_status(1, 0, 0, 1, 0, True, 0.0, 1.0)[0] == "FAIL"


def test_warning_can_be_caused_by_bbox_iou() -> None:
    status, reason, warnings = judge_status(1, 1, 1, 0, 0, True, 0.001, 0.98)

    assert status == "WARNING"
    assert "bbox_iou_min" in reason
    assert warnings


def test_final_status_precedence() -> None:
    rows = [
        argparse.Namespace(status="PASS"),
        argparse.Namespace(status="WARNING"),
        argparse.Namespace(status="FAIL"),
    ]
    assert final_status(rows) == "FAIL"
    assert exit_code_for_status("WARNING", fail_on_warning=False) == 0
    assert exit_code_for_status("WARNING", fail_on_warning=True) == 1


def test_final_status_warning_when_no_fail_or_error() -> None:
    rows = [argparse.Namespace(status="PASS"), argparse.Namespace(status="WARNING")]

    assert final_status(rows) == "WARNING"
    assert exit_code_for_status("FAIL", fail_on_warning=False) == 1
    assert exit_code_for_status("ERROR", fail_on_warning=False) == 1


def test_run_batch_writes_json_csv_and_mismatch_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_dir = tmp_path / "images"
    write_images(image_dir, ["pass.jpg", "warn.jpg", "fail.jpg"])
    args = make_args(tmp_path, image_dir)
    monkeypatch.setattr("scripts.compare_pytorch_onnx_batch.maybe_write_side_by_side", lambda *args, **kwargs: None)

    def run_pt(image_path: Path) -> TimedDetectionsResult:
        if image_path.stem == "fail":
            return timed_result([det()], total=20.0)
        return timed_result([det(0.90)], total=20.0)

    def run_onnx(image_path: Path) -> TimedDetectionsResult:
        if image_path.stem == "warn":
            return timed_result([det(0.88)], backend="onnx", total=10.0)
        if image_path.stem == "fail":
            return timed_result([], backend="onnx", total=10.0)
        return timed_result([det(0.895)], backend="onnx", total=10.0)

    result = run_batch(args, pytorch_runner=run_pt, onnx_runner=run_onnx, class_names={1: "short"})

    assert result.summary["final_status"] == "FAIL"
    assert result.summary["counts"]["pass_count"] == 1
    assert result.summary["counts"]["warning_count"] == 1
    assert result.summary["counts"]["fail_count"] == 1
    assert (args.output / "summary.json").exists()
    assert (args.output / "image_results.csv").exists()
    assert (args.output / "timing.csv").exists()
    assert "[WARNING] warn.jpg" in (args.output / "mismatch_images.txt").read_text(encoding="utf-8")
    assert "[FAIL] fail.jpg" in (args.output / "mismatch_images.txt").read_text(encoding="utf-8")

    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert summary["detection_summary"]["total_pytorch_only"] == 1

    with (args.output / "image_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["FAIL", "PASS", "WARNING"] or sorted(row["status"] for row in rows) == ["FAIL", "PASS", "WARNING"]


def test_timing_csv_contains_per_image_repeat_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_dir = tmp_path / "images"
    write_images(image_dir, ["one.jpg"])
    args = make_args(tmp_path, image_dir)
    monkeypatch.setattr("scripts.compare_pytorch_onnx_batch.maybe_write_side_by_side", lambda *args, **kwargs: None)

    run_batch(
        args,
        pytorch_runner=lambda _path: timed_result([], total=20.0),
        onnx_runner=lambda _path: timed_result([], backend="onnx", total=10.0),
        class_names={1: "short"},
    )

    with (args.output / "timing.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["backend"] for row in rows} == {"pytorch", "onnx"}
    assert {row["image_name"] for row in rows} == {"one.jpg"}


def test_run_batch_continues_after_image_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_dir = tmp_path / "images"
    write_images(image_dir, ["bad.jpg", "good.jpg"])
    args = make_args(tmp_path, image_dir)
    monkeypatch.setattr("scripts.compare_pytorch_onnx_batch.maybe_write_side_by_side", lambda *args, **kwargs: None)

    def run_pt(image_path: Path) -> TimedDetectionsResult:
        if image_path.stem == "bad":
            raise RuntimeError("boom")
        return timed_result([], total=20.0)

    def run_onnx(_image_path: Path) -> TimedDetectionsResult:
        return timed_result([], backend="onnx", total=10.0)

    result = run_batch(args, pytorch_runner=run_pt, onnx_runner=run_onnx, class_names={1: "short"})

    assert result.summary["final_status"] == "ERROR"
    assert result.summary["counts"]["error_count"] == 1
    assert result.summary["counts"]["pass_count"] == 1
    assert "[ERROR] bad.jpg" in (args.output / "error_images.txt").read_text(encoding="utf-8")


def test_error_result_uses_null_in_json_and_empty_csv_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_dir = tmp_path / "images"
    write_images(image_dir, ["bad.jpg"])
    args = make_args(tmp_path, image_dir)
    monkeypatch.setattr("scripts.compare_pytorch_onnx_batch.maybe_write_side_by_side", lambda *args, **kwargs: None)

    run_batch(
        args,
        pytorch_runner=lambda _path: (_ for _ in ()).throw(RuntimeError("boom")),
        onnx_runner=lambda _path: timed_result([], backend="onnx"),
        class_names={1: "short"},
    )

    per_image = json.loads((args.output / "per_image" / "bad.json").read_text(encoding="utf-8"))
    assert per_image["pytorch_detection_count"] is None

    with (args.output / "image_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["pytorch_detection_count"] == ""
    assert row["status"] == "ERROR"


def test_mismatch_and_error_text_formatters() -> None:
    warning = argparse.Namespace(
        status="WARNING",
        image_name="warn.jpg",
        confidence_diff_max=0.02,
        bbox_iou_min=0.995,
        pytorch_only_count=0,
        onnx_only_count=0,
        classes_match=True,
        status_reason="confidence",
    )
    error = argparse.Namespace(status="ERROR", image_name="bad.jpg", status_reason="failed", error_message="boom")

    assert "[WARNING] warn.jpg" in format_mismatch_text([warning])
    assert "[ERROR] bad.jpg" in format_error_text([error])


def test_fail_on_warning_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_dir = tmp_path / "images"
    write_images(image_dir, ["warn.jpg"])
    args = make_args(tmp_path, image_dir)
    args.fail_on_warning = True
    monkeypatch.setattr("scripts.compare_pytorch_onnx_batch.maybe_write_side_by_side", lambda *args, **kwargs: None)

    result = run_batch(
        args,
        pytorch_runner=lambda _path: timed_result([det(0.9)], total=20.0),
        onnx_runner=lambda _path: timed_result([det(0.87)], backend="onnx", total=10.0),
        class_names={1: "short"},
    )

    assert result.summary["final_status"] == "WARNING"
    assert result.exit_code == 1


def test_run_batch_returns_code_2_for_input_errors(tmp_path: Path) -> None:
    image_dir = tmp_path / "missing"
    args = make_args(tmp_path, tmp_path / "images")
    args.images = image_dir

    result = run_batch(args, pytorch_runner=lambda _path: timed_result([]), onnx_runner=lambda _path: timed_result([], "onnx"))

    assert result.exit_code == 2
