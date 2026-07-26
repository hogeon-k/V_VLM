from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import pytest

from model.defect_info import Detection
from scripts.compare_all_backends import (
    BACKEND_ORDER,
    BackendResult,
    BackendRunner,
    RunnerFactory,
    TimingRecord,
    collect_images,
    compare_detection_lists,
    run_benchmark,
    summarize_timing_records,
)


def detection(
    class_id: int = 2,
    confidence: float = 0.9,
    bbox: tuple[int, int, int, int] = (10, 10, 30, 30),
) -> Detection:
    names = {0: "open_circuit", 1: "short", 2: "missing_hole"}
    return Detection(
        class_id=class_id,
        class_name=names[class_id],
        confidence=confidence,
        x1=bbox[0],
        y1=bbox[1],
        x2=bbox[2],
        y2=bbox[3],
    )


def timing(
    image: str = "sample.jpg",
    backend: str = "pytorch",
    run_index: int = 0,
    is_warmup: bool = False,
    total_ms: float = 10.0,
    worker_pid: int | None = None,
    fallback_used: bool = False,
) -> TimingRecord:
    return TimingRecord(
        image=image,
        backend=backend,
        run_index=run_index,
        is_warmup=is_warmup,
        preprocess_ms=1.0,
        inference_ms=6.0,
        postprocess_ms=2.0,
        backend_total_ms=9.0,
        host_roundtrip_ms=total_ms,
        end_to_end_ms=total_ms,
        worker_pid=worker_pid,
        fallback_used=fallback_used,
    )


class FakeRunner(BackendRunner):
    def __init__(
        self,
        name: str,
        precision: str,
        detections: list[Detection],
        provider: str,
        *,
        fail: bool = False,
        worker_pid: int | None = None,
        fallback_used: bool = False,
    ) -> None:
        self.name = name
        self.precision = precision
        self.values = detections
        self.provider = provider
        self.fail = fail
        self.worker_pid = worker_pid
        self.fallback_used = fallback_used
        self.closed = False
        self.startup_ms = 25.0 if name.startswith("tensorrt") else 5.0

    def infer(self, image_path: Path, run_index: int, is_warmup: bool) -> BackendResult:
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return BackendResult(
            backend=self.name,
            detections=self.values,
            timing=timing(
                image_path.name,
                self.name,
                run_index,
                is_warmup,
                total_ms=10.0 + run_index,
                worker_pid=self.worker_pid,
                fallback_used=self.fallback_used,
            ),
            provider=self.provider,
            precision=self.precision,
        )

    def close(self) -> None:
        self.closed = True


def make_args(tmp_path: Path, **overrides: object) -> Namespace:
    images = tmp_path / "images"
    images.mkdir()
    (images / "sample.jpg").write_bytes(b"image")
    paths = {
        "pytorch_model": tmp_path / "best.pt",
        "onnx_model": tmp_path / "best.onnx",
        "tensorrt_fp32_engine": tmp_path / "best_fp32.engine",
        "tensorrt_fp16_engine": tmp_path / "best_fp16.engine",
        "metadata": tmp_path / "metadata.json",
        "tensorrt_executable": tmp_path / "infer.exe",
    }
    for path in paths.values():
        path.write_bytes(b"x")
    values: dict[str, object] = {
        "images": images,
        **paths,
        "output": tmp_path / "output",
        "imgsz": 960,
        "conf": 0.15,
        "iou": 0.5,
        "match_iou": 0.5,
        "warmup": 1,
        "repeat": 2,
        "device": "0",
        "provider": "CUDAExecutionProvider",
        "extensions": ".jpg,.png",
        "max_images": None,
        "recursive": False,
        "fail_on_mismatch": False,
    }
    values.update(overrides)
    return Namespace(**values)


def make_factories(
    *,
    target_detections: dict[str, list[Detection]] | None = None,
    failing_backend: str | None = None,
    fallback_backend: str | None = None,
) -> tuple[list[RunnerFactory], dict[str, FakeRunner]]:
    target_detections = target_detections or {}
    providers = {
        "pytorch": "cuda:0",
        "onnx_cuda": "CUDAExecutionProvider",
        "tensorrt_fp32": "Native TensorRT",
        "tensorrt_fp16": "Native TensorRT",
    }
    precision = {
        "pytorch": "FP32",
        "onnx_cuda": "FP32",
        "tensorrt_fp32": "FP32",
        "tensorrt_fp16": "FP16",
    }
    runners: dict[str, FakeRunner] = {}
    factories: list[RunnerFactory] = []
    for index, name in enumerate(BACKEND_ORDER):
        runner = FakeRunner(
            name,
            precision[name],
            target_detections.get(name, [detection()]),
            providers[name],
            fail=name == failing_backend,
            worker_pid=(7000 + index if name.startswith("tensorrt") else None),
            fallback_used=name == fallback_backend,
        )
        runners[name] = runner
        factories.append(
            RunnerFactory(
                name,
                precision[name],
                lambda runner=runner: runner,
            )
        )
    return factories, runners


def test_collect_images_filters_extensions_and_supports_unicode(tmp_path) -> None:
    image_dir = tmp_path / "한글 폴더"
    image_dir.mkdir()
    (image_dir / "가.JPG").write_bytes(b"x")
    (image_dir / "b.png").write_bytes(b"x")
    (image_dir / "ignore.txt").write_bytes(b"x")

    images = collect_images(image_dir, (".jpg",), max_images=1)

    assert images == [image_dir / "가.JPG"]


def test_matching_reports_exact_match() -> None:
    summary, rows = compare_detection_lists(
        "a.jpg", "pytorch", "onnx_cuda", [detection()], [detection()], 0.5
    )

    assert summary.status == "PASS"
    assert summary.matched == 1
    assert summary.mismatch_count == 0
    assert rows[0].match_status == "MATCHED"


def test_matching_reports_false_positive_and_false_negative() -> None:
    summary, rows = compare_detection_lists(
        "a.jpg",
        "pytorch",
        "onnx_cuda",
        [detection(bbox=(0, 0, 10, 10))],
        [detection(bbox=(30, 30, 40, 40))],
        0.5,
    )

    assert summary.status == "FAIL"
    assert summary.false_positive == 1
    assert summary.false_negative == 1
    assert {row.match_status for row in rows} == {
        "FALSE_POSITIVE",
        "FALSE_NEGATIVE",
    }


def test_matching_reports_class_mismatch() -> None:
    summary, rows = compare_detection_lists(
        "a.jpg",
        "pytorch",
        "onnx_cuda",
        [detection(class_id=2)],
        [detection(class_id=1)],
        0.5,
    )

    assert summary.class_mismatch == 1
    assert summary.status == "FAIL"
    assert rows[0].match_status == "CLASS_MISMATCH"


def test_matching_records_confidence_and_bbox_warning() -> None:
    summary, rows = compare_detection_lists(
        "a.jpg",
        "pytorch",
        "onnx_cuda",
        [detection(confidence=0.9)],
        [detection(confidence=0.8, bbox=(11, 10, 31, 30))],
        0.5,
    )

    assert summary.status == "WARNING"
    assert summary.confidence_delta_max == pytest.approx(0.1)
    assert summary.bbox_iou_min is not None
    assert rows[0].match_status == "WARNING"


def test_timing_statistics_exclude_warmup_and_calculate_percentiles() -> None:
    rows = [
        timing(is_warmup=True, total_ms=1000.0),
        timing(run_index=0, total_ms=10.0),
        timing(run_index=1, total_ms=20.0),
    ]

    summary = summarize_timing_records(rows)

    assert summary["end_to_end_ms"]["count"] == 2
    assert summary["end_to_end_ms"]["mean_ms"] == pytest.approx(15.0)
    assert summary["end_to_end_ms"]["median_ms"] == pytest.approx(15.0)
    assert summary["end_to_end_ms"]["p95_ms"] == pytest.approx(19.5)
    assert summary["end_to_end_ms"]["fps"] == pytest.approx(1000 / 15)


def test_run_benchmark_writes_required_reports_and_closes_all_runners(tmp_path) -> None:
    args = make_args(tmp_path)
    factories, runners = make_factories()

    run = run_benchmark(args, factories)

    assert run.summary["final_status"] == "PASS"
    assert run.exit_code == 0
    assert all(runner.closed for runner in runners.values())
    for filename in (
        "summary.json",
        "summary.csv",
        "per_image_results.csv",
        "detection_comparisons.csv",
        "timing_samples.csv",
        "report.md",
    ):
        assert (args.output / filename).is_file()


def test_summary_csv_has_required_columns_and_json_is_serializable(tmp_path) -> None:
    args = make_args(tmp_path)
    factories, _ = make_factories()

    run_benchmark(args, factories)

    with (args.output / "summary.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 4
    assert {
        "backend",
        "provider",
        "precision",
        "startup_ms",
        "first_request_ms",
        "inference_mean_ms",
        "end_to_end_mean_ms",
        "p95_ms",
        "throughput_qps",
        "fallback_count",
        "mismatch_count",
        "result",
    }.issubset(rows[0])
    json.dumps(payload)


def test_backend_failure_is_recorded_and_other_backends_continue(tmp_path) -> None:
    args = make_args(tmp_path)
    factories, runners = make_factories(failing_backend="tensorrt_fp32")

    run = run_benchmark(args, factories)

    rows = {row["backend"]: row for row in run.backend_rows}
    assert rows["tensorrt_fp32"]["result"] == "FAIL"
    assert rows["tensorrt_fp16"]["image_count"] == 1
    assert runners["tensorrt_fp32"].closed is True
    assert runners["tensorrt_fp16"].closed is True


def test_backend_initialization_failure_does_not_stop_later_backend(tmp_path) -> None:
    args = make_args(tmp_path)
    factories, runners = make_factories()

    def fail_create() -> BackendRunner:
        raise FileNotFoundError("FP32 engine missing")

    factories[2] = RunnerFactory("tensorrt_fp32", "FP32", fail_create)

    run = run_benchmark(args, factories)

    assert "FP32 engine missing" in run.summary["backend_errors"]["tensorrt_fp32"]
    assert runners["tensorrt_fp16"].closed is True
    assert next(
        row for row in run.backend_rows if row["backend"] == "tensorrt_fp32"
    )["result"] == "FAIL"


def test_baseline_failure_stops_target_execution(tmp_path) -> None:
    args = make_args(tmp_path)
    factories, runners = make_factories(failing_backend="pytorch")

    run = run_benchmark(args, factories)

    assert run.summary["final_status"] == "FAIL"
    assert runners["pytorch"].closed is True
    assert runners["onnx_cuda"].closed is False
    assert "Not run because" in run.summary["backend_errors"]["onnx_cuda"]


def test_fail_on_mismatch_returns_exit_code_one_and_writes_cases(tmp_path) -> None:
    args = make_args(tmp_path, fail_on_mismatch=True)
    factories, _ = make_factories(
        target_detections={"onnx_cuda": []}
    )

    run = run_benchmark(args, factories)

    assert run.exit_code == 1
    assert run.summary["final_status"] == "FAIL"
    case_dir = args.output / "mismatch_cases" / "sample"
    assert (case_dir / "pytorch.json").is_file()
    assert (case_dir / "onnx_cuda.json").is_file()


def test_fallback_is_reported_as_warning(tmp_path) -> None:
    args = make_args(tmp_path)
    factories, _ = make_factories(fallback_backend="tensorrt_fp16")

    run = run_benchmark(args, factories)

    row = next(
        item for item in run.backend_rows if item["backend"] == "tensorrt_fp16"
    )
    assert row["fallback_count"] == args.repeat
    assert row["result"] == "WARNING"


def test_tensor_worker_pid_is_reused_across_repeats(tmp_path) -> None:
    args = make_args(tmp_path, repeat=3)
    factories, _ = make_factories()

    run = run_benchmark(args, factories)

    rows = {
        row["backend"]: row
        for row in run.backend_rows
        if row["backend"].startswith("tensorrt")
    }
    assert all(row["worker_pid_reused"] is True for row in rows.values())
    assert all(len(row["worker_pids"]) == 1 for row in rows.values())


def test_onnx_cpu_fallback_is_warning(tmp_path) -> None:
    args = make_args(tmp_path)
    factories, runners = make_factories()
    runners["onnx_cuda"].provider = "CPUExecutionProvider"

    run = run_benchmark(args, factories)

    row = next(item for item in run.backend_rows if item["backend"] == "onnx_cuda")
    assert row["provider"] == "CPUExecutionProvider"
    assert row["result"] == "WARNING"


def test_report_explains_timing_and_final_status(tmp_path) -> None:
    args = make_args(tmp_path)
    factories, _ = make_factories()

    run_benchmark(args, factories)

    report = (args.output / "report.md").read_text(encoding="utf-8")
    assert "Backend stage timings are backend-reported values" in report
    assert "TensorRT startup is reported separately" in report
    assert "Final status: **PASS**" in report


def test_mismatch_cases_are_refreshed_between_runs(tmp_path) -> None:
    args = make_args(tmp_path)
    mismatch_factories, _ = make_factories(
        target_detections={"onnx_cuda": []}
    )
    pass_factories, _ = make_factories()

    run_benchmark(args, mismatch_factories)
    assert (args.output / "mismatch_cases").is_dir()

    run_benchmark(args, pass_factories)
    assert not (args.output / "mismatch_cases").exists()
