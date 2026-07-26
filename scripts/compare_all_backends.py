from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import shutil
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.defect_info import Detection
from scripts.compare_pytorch_onnx import (
    DetectionMatch,
    load_class_names,
    match_detections,
    run_pytorch_once,
)
from scripts.compare_pytorch_onnx_batch import (
    SUPPORTED_EXTENSIONS,
    extensions_from_arg,
    pytorch_actual_device,
    timing_stats,
)
from service.onnx_detector import OnnxDetector, detection_to_dict
from service.tensorrt_detector_adapter import TensorRtDetectorAdapter

CSV_ENCODING = "utf-8-sig"
PASS_CONFIDENCE_DELTA_MAX = 0.01
PASS_BBOX_IOU_MIN = 0.99
BACKEND_ORDER = ("pytorch", "onnx_cuda", "tensorrt_fp32", "tensorrt_fp16")
COMPARISON_PAIRS = (
    ("pytorch", "onnx_cuda"),
    ("pytorch", "tensorrt_fp32"),
    ("pytorch", "tensorrt_fp16"),
    ("tensorrt_fp32", "tensorrt_fp16"),
)


@dataclass(frozen=True, slots=True)
class TimingRecord:
    image: str
    backend: str
    run_index: int
    is_warmup: bool
    preprocess_ms: float | None
    inference_ms: float | None
    postprocess_ms: float | None
    backend_total_ms: float | None
    host_roundtrip_ms: float
    end_to_end_ms: float
    worker_pid: int | None = None
    fallback_used: bool = False


@dataclass(frozen=True, slots=True)
class BackendResult:
    backend: str
    detections: list[Detection]
    timing: TimingRecord
    provider: str
    precision: str


@dataclass(slots=True)
class ImageBackendResult:
    image: str
    image_path: str
    backend: str
    provider: str
    precision: str
    status: str
    detection_count: int | None = None
    class_names: list[str] = field(default_factory=list)
    mean_confidence: float | None = None
    preprocess_ms: float | None = None
    inference_ms: float | None = None
    postprocess_ms: float | None = None
    backend_total_ms: float | None = None
    end_to_end_ms: float | None = None
    error: str = ""
    detections: list[dict[str, Any]] = field(default_factory=list)
    stability_mismatch_count: int = 0


@dataclass(frozen=True, slots=True)
class DetectionComparison:
    image: str
    reference_backend: str
    target_backend: str
    reference_index: int | None
    target_index: int | None
    reference_class: str
    target_class: str
    reference_confidence: float | None
    target_confidence: float | None
    confidence_delta: float | None
    bbox_iou: float | None
    match_status: str
    reason: str


@dataclass(frozen=True, slots=True)
class ComparisonSummary:
    image: str
    reference_backend: str
    target_backend: str
    status: str
    matched: int
    false_positive: int
    false_negative: int
    class_mismatch: int
    confidence_delta_mean: float | None
    confidence_delta_max: float | None
    bbox_iou_mean: float | None
    bbox_iou_min: float | None
    mismatch_count: int
    reason: str


@dataclass(frozen=True, slots=True)
class RunnerFactory:
    name: str
    precision: str
    create: Callable[[], "BackendRunner"]


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    summary: dict[str, Any]
    backend_rows: list[dict[str, Any]]
    image_rows: list[ImageBackendResult]
    comparisons: list[ComparisonSummary]
    detection_rows: list[DetectionComparison]
    timing_rows: list[TimingRecord]
    exit_code: int


class BackendRunner(ABC):
    name: str
    precision: str
    startup_ms: float | None = None

    @abstractmethod
    def infer(self, image_path: Path, run_index: int, is_warmup: bool) -> BackendResult:
        raise NotImplementedError

    def warmup(self, image_path: Path, count: int) -> list[BackendResult]:
        return [self.infer(image_path, index, True) for index in range(count)]

    def close(self) -> None:
        return None


class PyTorchBackendRunner(BackendRunner):
    name = "pytorch"
    precision = "FP32"

    def __init__(
        self,
        model_path: Path,
        imgsz: int,
        conf: float,
        iou: float,
        device: str,
    ) -> None:
        from ultralytics import YOLO

        started = time.perf_counter()
        self.model = YOLO(str(model_path))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.startup_ms = (time.perf_counter() - started) * 1000

    def infer(self, image_path: Path, run_index: int, is_warmup: bool) -> BackendResult:
        started = time.perf_counter()
        detections, values = run_pytorch_once(
            self.model,
            image_path,
            self.imgsz,
            self.conf,
            self.iou,
            self.device,
        )
        host_ms = (time.perf_counter() - started) * 1000
        provider = pytorch_actual_device(self.model)
        return BackendResult(
            backend=self.name,
            detections=detections,
            timing=_timing_record(
                image_path,
                self.name,
                run_index,
                is_warmup,
                values,
                host_ms,
            ),
            provider=provider,
            precision=self.precision,
        )

    def close(self) -> None:
        self.model = None
        _release_gpu_memory()


class OnnxBackendRunner(BackendRunner):
    name = "onnx_cuda"
    precision = "FP32"

    def __init__(
        self,
        model_path: Path,
        imgsz: int,
        conf: float,
        iou: float,
        provider: str,
        class_names: dict[int, str],
    ) -> None:
        started = time.perf_counter()
        self.detector = OnnxDetector(
            model_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            class_names=class_names,
            requested_provider=provider,
            require_cuda=False,
        )
        self.detector._load_session()
        self.startup_ms = (time.perf_counter() - started) * 1000

    def infer(self, image_path: Path, run_index: int, is_warmup: bool) -> BackendResult:
        started = time.perf_counter()
        result = self.detector.detect_timed(image_path)
        host_ms = (time.perf_counter() - started) * 1000
        values = {
            "preprocess_ms": result.preprocess_ms,
            "inference_ms": result.inference_ms,
            "postprocess_ms": result.postprocess_ms,
            "total_ms": result.total_ms,
        }
        provider = result.providers[0] if result.providers else ""
        return BackendResult(
            backend=self.name,
            detections=result.detections,
            timing=_timing_record(
                image_path,
                self.name,
                run_index,
                is_warmup,
                values,
                host_ms,
            ),
            provider=provider,
            precision=self.precision,
        )

    def close(self) -> None:
        self.detector._session = None
        _release_gpu_memory()


class TensorRtBackendRunner(BackendRunner):
    def __init__(
        self,
        name: str,
        precision: str,
        executable_path: Path,
        engine_path: Path,
        metadata_path: Path,
        imgsz: int,
        conf: float,
        iou: float,
        device_id: int,
    ) -> None:
        self.name = name
        self.precision = precision
        self._output_context = tempfile.TemporaryDirectory(
            prefix=f"vvlm_{name}_comparison_"
        )
        self._output_dir = Path(self._output_context.name)
        self.adapter = TensorRtDetectorAdapter(
            executable_path=executable_path,
            engine_path=engine_path,
            metadata_path=metadata_path,
            device_id=device_id,
            image_size=imgsz,
            confidence_threshold=conf,
            iou_threshold=iou,
            engine_label=precision.lower(),
            use_persistent_worker=True,
            fallback_to_oneshot=True,
            generate_annotated_image=False,
        )
        started = time.perf_counter()
        worker = self.adapter._get_persistent_worker()
        worker.start()
        self.startup_ms = (
            worker.startup_ms
            if worker.startup_ms is not None
            else (time.perf_counter() - started) * 1000
        )

    def infer(self, image_path: Path, run_index: int, is_warmup: bool) -> BackendResult:
        output_path = self._output_dir / f"{image_path.stem}_{run_index}.jpg"
        started = time.perf_counter()
        result = self.adapter.detect(image_path, output_path=output_path)
        host_ms = (time.perf_counter() - started) * 1000
        metadata = self.adapter.last_metadata
        if metadata is None:
            raise RuntimeError("TensorRT adapter did not provide run metadata.")
        values = {
            "preprocess_ms": metadata.timing_ms.get("preprocess"),
            "inference_ms": metadata.timing_ms.get("inference"),
            "postprocess_ms": metadata.timing_ms.get("postprocess"),
            "total_ms": metadata.timing_ms.get("total"),
        }
        worker = self.adapter._persistent_worker
        return BackendResult(
            backend=self.name,
            detections=result.detections,
            timing=_timing_record(
                image_path,
                self.name,
                run_index,
                is_warmup,
                values,
                host_ms,
                worker_pid=worker.pid if worker is not None else None,
                fallback_used=metadata.execution_mode != "persistent",
            ),
            provider="Native TensorRT",
            precision=self.precision,
        )

    def close(self) -> None:
        try:
            self.adapter.close()
        finally:
            self._output_context.cleanup()
            _release_gpu_memory()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PyTorch CUDA, Python ONNX Runtime CUDA, and persistent "
            "TensorRT FP32/FP16 inference under common settings."
        )
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--pytorch-model", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--tensorrt-fp32-engine", type=Path, required=True)
    parser.add_argument("--tensorrt-fp16-engine", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tensorrt-executable",
        type=Path,
        default=Path("cpp_inference/build_gpu/Release/pcb_onnx_infer.exe"),
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--extensions", default=",".join(SUPPORTED_EXTENSIONS))
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args(argv)


def collect_images(
    directory: Path,
    extensions: tuple[str, ...],
    recursive: bool = False,
    max_images: int | None = None,
) -> list[Path]:
    normalized = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    images = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in normalized
    )
    return images if max_images is None else images[:max_images]


def validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not args.images.is_dir():
        errors.append(f"Image directory does not exist: {args.images}")
    if not args.pytorch_model.is_file():
        errors.append(f"PyTorch model does not exist: {args.pytorch_model}")
    if args.imgsz <= 0:
        errors.append("--imgsz must be > 0")
    if args.warmup < 0:
        errors.append("--warmup must be >= 0")
    if args.repeat < 1:
        errors.append("--repeat must be >= 1")
    if args.max_images is not None and args.max_images < 1:
        errors.append("--max-images must be >= 1")
    for name in ("conf", "iou", "match_iou"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            errors.append(f"--{name.replace('_', '-')} must be between 0 and 1")
    return errors


def build_runner_factories(
    args: argparse.Namespace,
    class_names: dict[int, str],
) -> list[RunnerFactory]:
    device_id = int(args.device) if str(args.device).isdigit() else 0
    return [
        RunnerFactory(
            "pytorch",
            "FP32",
            lambda: PyTorchBackendRunner(
                args.pytorch_model,
                args.imgsz,
                args.conf,
                args.iou,
                str(args.device),
            ),
        ),
        RunnerFactory(
            "onnx_cuda",
            "FP32",
            lambda: OnnxBackendRunner(
                args.onnx_model,
                args.imgsz,
                args.conf,
                args.iou,
                args.provider,
                class_names,
            ),
        ),
        RunnerFactory(
            "tensorrt_fp32",
            "FP32",
            lambda: TensorRtBackendRunner(
                "tensorrt_fp32",
                "FP32",
                args.tensorrt_executable,
                args.tensorrt_fp32_engine,
                args.metadata,
                args.imgsz,
                args.conf,
                args.iou,
                device_id,
            ),
        ),
        RunnerFactory(
            "tensorrt_fp16",
            "FP16",
            lambda: TensorRtBackendRunner(
                "tensorrt_fp16",
                "FP16",
                args.tensorrt_executable,
                args.tensorrt_fp16_engine,
                args.metadata,
                args.imgsz,
                args.conf,
                args.iou,
                device_id,
            ),
        ),
    ]


def run_benchmark(
    args: argparse.Namespace,
    runner_factories: list[RunnerFactory] | None = None,
) -> BenchmarkRun:
    validation_errors = validate_args(args)
    if validation_errors:
        raise ValueError("\n".join(validation_errors))
    images = collect_images(
        args.images,
        extensions_from_arg(args.extensions),
        args.recursive,
        args.max_images,
    )
    if not images:
        raise ValueError(f"No supported images found in: {args.images}")

    args.output.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(args.metadata)
    factories = runner_factories or build_runner_factories(args, class_names)
    image_rows: list[ImageBackendResult] = []
    timing_rows: list[TimingRecord] = []
    canonical: dict[tuple[str, str], list[Detection]] = {}
    startup_ms: dict[str, float | None] = {}
    backend_errors: dict[str, str] = {}
    backend_metadata: dict[str, dict[str, Any]] = {}

    for factory in factories:
        runner: BackendRunner | None = None
        print(f"[{factory.name}] initializing")
        try:
            runner = factory.create()
            startup_ms[factory.name] = runner.startup_ms
            warmup_results = runner.warmup(images[0], args.warmup)
            timing_rows.extend(result.timing for result in warmup_results)

            for image_index, image_path in enumerate(images, start=1):
                measured: list[BackendResult] = []
                try:
                    for run_index in range(args.repeat):
                        result = runner.infer(image_path, run_index, False)
                        measured.append(result)
                        timing_rows.append(result.timing)
                    first = measured[0]
                    canonical[(factory.name, str(image_path))] = first.detections
                    stability_mismatches = sum(
                        _comparison_has_mismatch(
                            compare_detection_lists(
                                image_path.name,
                                factory.name,
                                factory.name,
                                first.detections,
                                repeat_result.detections,
                                args.match_iou,
                            )[0]
                        )
                        for repeat_result in measured[1:]
                    )
                    image_rows.append(
                        build_image_row(
                            image_path,
                            first,
                            measured,
                            stability_mismatches,
                        )
                    )
                    print(
                        f"[{factory.name}] {image_index}/{len(images)} "
                        f"{image_path.name}: detections={len(first.detections)}"
                    )
                except Exception as exc:
                    image_rows.append(
                        ImageBackendResult(
                            image=image_path.name,
                            image_path=str(image_path),
                            backend=factory.name,
                            provider="",
                            precision=factory.precision,
                            status="FAIL",
                            error=str(exc),
                        )
                    )
                    print(f"[{factory.name}] {image_path.name}: ERROR {exc}")
            backend_metadata[factory.name] = _runner_metadata(runner)
        except Exception as exc:
            startup_ms[factory.name] = None
            backend_errors[factory.name] = str(exc)
            backend_metadata[factory.name] = {}
            print(f"[{factory.name}] initialization failed: {exc}")
            if factory.name == "pytorch":
                break
        finally:
            if runner is not None:
                runner.close()
            _release_gpu_memory()

    if "pytorch" in backend_errors:
        for factory in factories:
            if factory.name != "pytorch" and factory.name not in backend_errors:
                backend_errors[factory.name] = (
                    "Not run because the PyTorch baseline failed."
                )

    comparisons: list[ComparisonSummary] = []
    detection_rows: list[DetectionComparison] = []
    if "pytorch" not in backend_errors:
        for reference_backend, target_backend in COMPARISON_PAIRS:
            for image_path in images:
                reference = canonical.get((reference_backend, str(image_path)))
                target = canonical.get((target_backend, str(image_path)))
                if reference is None or target is None:
                    comparisons.append(
                        ComparisonSummary(
                            image=image_path.name,
                            reference_backend=reference_backend,
                            target_backend=target_backend,
                            status="FAIL",
                            matched=0,
                            false_positive=0,
                            false_negative=0,
                            class_mismatch=0,
                            confidence_delta_mean=None,
                            confidence_delta_max=None,
                            bbox_iou_mean=None,
                            bbox_iou_min=None,
                            mismatch_count=1,
                            reason="Backend result unavailable.",
                        )
                    )
                    continue
                summary, rows = compare_detection_lists(
                    image_path.name,
                    reference_backend,
                    target_backend,
                    reference,
                    target,
                    args.match_iou,
                )
                comparisons.append(summary)
                detection_rows.extend(rows)

    backend_rows = build_backend_rows(
        factories,
        images,
        image_rows,
        timing_rows,
        comparisons,
        startup_ms,
        backend_errors,
        backend_metadata,
        args,
    )
    final_status = _overall_status(
        [str(row["result"]) for row in backend_rows]
        + [comparison.status for comparison in comparisons]
    )
    summary = {
        "final_status": final_status,
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "configuration": {
            "images": str(args.images),
            "image_count": len(images),
            "pytorch_model": str(args.pytorch_model),
            "onnx_model": str(args.onnx_model),
            "tensorrt_fp32_engine": str(args.tensorrt_fp32_engine),
            "tensorrt_fp16_engine": str(args.tensorrt_fp16_engine),
            "metadata": str(args.metadata),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "match_iou": args.match_iou,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "device": str(args.device),
            "provider": args.provider,
            "warmup_policy": "Warm up each backend on the first selected image only.",
            "timing_policy": (
                "Backend stage timings are reported separately from Python host "
                "roundtrip and end-to-end wall time."
            ),
        },
        "backend_summaries": backend_rows,
        "comparison_summaries": [asdict(item) for item in comparisons],
        "backend_errors": backend_errors,
    }
    exit_code = (
        1
        if args.fail_on_mismatch
        and (
            final_status == "FAIL"
            or any(comparison.status == "FAIL" for comparison in comparisons)
        )
        else 0
    )
    run = BenchmarkRun(
        summary=summary,
        backend_rows=backend_rows,
        image_rows=image_rows,
        comparisons=comparisons,
        detection_rows=detection_rows,
        timing_rows=timing_rows,
        exit_code=exit_code,
    )
    write_reports(args.output, run, canonical)
    return run


def build_image_row(
    image_path: Path,
    first: BackendResult,
    measured: list[BackendResult],
    stability_mismatches: int,
) -> ImageBackendResult:
    timing = summarize_timing_records([result.timing for result in measured])
    confidences = [detection.confidence for detection in first.detections]
    provider_fallback = (
        first.backend == "onnx_cuda"
        and first.provider != "CUDAExecutionProvider"
    )
    fallback_used = any(result.timing.fallback_used for result in measured)
    status = "WARNING" if stability_mismatches or provider_fallback or fallback_used else "PASS"
    return ImageBackendResult(
        image=image_path.name,
        image_path=str(image_path),
        backend=first.backend,
        provider=first.provider,
        precision=first.precision,
        status=status,
        detection_count=len(first.detections),
        class_names=[detection.class_name for detection in first.detections],
        mean_confidence=(
            sum(confidences) / len(confidences) if confidences else None
        ),
        preprocess_ms=_mean_stat(timing, "preprocess_ms"),
        inference_ms=_mean_stat(timing, "inference_ms"),
        postprocess_ms=_mean_stat(timing, "postprocess_ms"),
        backend_total_ms=_mean_stat(timing, "backend_total_ms"),
        end_to_end_ms=_mean_stat(timing, "end_to_end_ms"),
        detections=[detection_to_dict(item) for item in first.detections],
        stability_mismatch_count=stability_mismatches,
    )


def compare_detection_lists(
    image: str,
    reference_backend: str,
    target_backend: str,
    reference: list[Detection],
    target: list[Detection],
    match_iou: float,
) -> tuple[ComparisonSummary, list[DetectionComparison]]:
    matches = match_detections(reference, target, match_iou=match_iou)
    matched = [
        match
        for match in matches
        if match.status == "MATCHED"
        and match.pt is not None
        and match.onnx is not None
    ]
    reference_only = [
        match for match in matches if match.status == "PT_ONLY" and match.pt is not None
    ]
    target_only = [
        match
        for match in matches
        if match.status == "ONNX_ONLY" and match.onnx is not None
    ]
    class_mismatches = _pair_class_mismatches(
        reference_only,
        target_only,
        match_iou,
    )
    mismatched_reference_ids = {id(left.pt) for left, _, _ in class_mismatches}
    mismatched_target_ids = {id(right.onnx) for _, right, _ in class_mismatches}
    reference_only = [
        match for match in reference_only if id(match.pt) not in mismatched_reference_ids
    ]
    target_only = [
        match for match in target_only if id(match.onnx) not in mismatched_target_ids
    ]

    rows: list[DetectionComparison] = []
    confidence_deltas: list[float] = []
    bbox_ious: list[float] = []
    for match in matched:
        assert match.pt is not None and match.onnx is not None
        reference_index = reference.index(match.pt)
        target_index = target.index(match.onnx)
        delta = abs(match.pt.confidence - match.onnx.confidence)
        confidence_deltas.append(delta)
        bbox_ious.append(match.iou)
        warning = delta > PASS_CONFIDENCE_DELTA_MAX or match.iou < PASS_BBOX_IOU_MIN
        rows.append(
            _detection_comparison(
                image,
                reference_backend,
                target_backend,
                reference_index,
                target_index,
                match.pt,
                match.onnx,
                delta,
                match.iou,
                "WARNING" if warning else "MATCHED",
                (
                    "Confidence or bbox differs beyond the strict comparison tolerance."
                    if warning
                    else ""
                ),
            )
        )
    for left, right, overlap in class_mismatches:
        assert left.pt is not None and right.onnx is not None
        rows.append(
            _detection_comparison(
                image,
                reference_backend,
                target_backend,
                reference.index(left.pt),
                target.index(right.onnx),
                left.pt,
                right.onnx,
                abs(left.pt.confidence - right.onnx.confidence),
                overlap,
                "CLASS_MISMATCH",
                "Overlapping detections have different classes.",
            )
        )
    for match in reference_only:
        assert match.pt is not None
        rows.append(
            _detection_comparison(
                image,
                reference_backend,
                target_backend,
                reference.index(match.pt),
                None,
                match.pt,
                None,
                None,
                None,
                "FALSE_NEGATIVE",
                "Reference detection is missing from target backend.",
            )
        )
    for match in target_only:
        assert match.onnx is not None
        rows.append(
            _detection_comparison(
                image,
                reference_backend,
                target_backend,
                None,
                target.index(match.onnx),
                None,
                match.onnx,
                None,
                None,
                "FALSE_POSITIVE",
                "Target backend returned an additional detection.",
            )
        )

    mismatch_count = len(reference_only) + len(target_only) + len(class_mismatches)
    has_warning = any(row.match_status == "WARNING" for row in rows)
    status = "FAIL" if mismatch_count else ("WARNING" if has_warning else "PASS")
    reason = (
        f"FP={len(target_only)}, FN={len(reference_only)}, "
        f"class_mismatch={len(class_mismatches)}"
        if mismatch_count
        else (
            "Detection counts/classes match; strict confidence or bbox tolerance exceeded."
            if has_warning
            else "Detection counts, classes, confidence, and boxes match."
        )
    )
    return (
        ComparisonSummary(
            image=image,
            reference_backend=reference_backend,
            target_backend=target_backend,
            status=status,
            matched=len(matched),
            false_positive=len(target_only),
            false_negative=len(reference_only),
            class_mismatch=len(class_mismatches),
            confidence_delta_mean=(
                sum(confidence_deltas) / len(confidence_deltas)
                if confidence_deltas
                else None
            ),
            confidence_delta_max=max(confidence_deltas) if confidence_deltas else None,
            bbox_iou_mean=(
                sum(bbox_ious) / len(bbox_ious) if bbox_ious else None
            ),
            bbox_iou_min=min(bbox_ious) if bbox_ious else None,
            mismatch_count=mismatch_count,
            reason=reason,
        ),
        rows,
    )


def summarize_timing_records(
    rows: list[TimingRecord],
) -> dict[str, dict[str, float | None]]:
    measured = [row for row in rows if not row.is_warmup]
    return {
        field_name: timing_stats(
            [
                float(value)
                for row in measured
                if (value := getattr(row, field_name)) is not None
            ]
        )
        for field_name in (
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "backend_total_ms",
            "host_roundtrip_ms",
            "end_to_end_ms",
        )
    }


def build_backend_rows(
    factories: list[RunnerFactory],
    images: list[Path],
    image_rows: list[ImageBackendResult],
    timing_rows: list[TimingRecord],
    comparisons: list[ComparisonSummary],
    startup_ms: dict[str, float | None],
    backend_errors: dict[str, str],
    backend_metadata: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factory in factories:
        backend_images = [row for row in image_rows if row.backend == factory.name]
        backend_timings = [
            row
            for row in timing_rows
            if row.backend == factory.name and not row.is_warmup
        ]
        timing = summarize_timing_records(backend_timings)
        related_comparisons = [
            item for item in comparisons if item.target_backend == factory.name
        ]
        fallback_count = sum(row.fallback_used for row in backend_timings)
        error_count = sum(row.status == "FAIL" for row in backend_images)
        mismatch_count = sum(item.mismatch_count for item in related_comparisons)
        provider = next(
            (row.provider for row in backend_images if row.provider),
            "",
        )
        status_candidates = [
            row.status for row in backend_images
        ] + [item.status for item in related_comparisons]
        worker_pids = sorted(
            {
                row.worker_pid
                for row in backend_timings
                if row.worker_pid is not None
            }
        )
        worker_pid_reused = (
            len(worker_pids) == 1
            if factory.name.startswith("tensorrt") and backend_timings
            else None
        )
        if factory.name in backend_errors or error_count:
            result = "FAIL"
        elif factory.name == "pytorch":
            result = "BASELINE"
        else:
            result = _overall_status(status_candidates)
            if (
                factory.name == "onnx_cuda"
                and provider != "CUDAExecutionProvider"
                and result == "PASS"
            ):
                result = "WARNING"
            if fallback_count and result == "PASS":
                result = "WARNING"
        end_to_end = timing["end_to_end_ms"]
        rows.append(
            {
                "backend": factory.name,
                "provider": provider,
                "precision": factory.precision,
                "image_count": len(backend_images),
                "warmup": args.warmup,
                "repeat": args.repeat,
                "detection_count": sum(
                    row.detection_count or 0 for row in backend_images
                ),
                "startup_ms": startup_ms.get(factory.name),
                "first_request_ms": (
                    backend_timings[0].end_to_end_ms if backend_timings else None
                ),
                "preprocess_mean_ms": _mean_stat(timing, "preprocess_ms"),
                "inference_mean_ms": _mean_stat(timing, "inference_ms"),
                "postprocess_mean_ms": _mean_stat(timing, "postprocess_ms"),
                "backend_total_mean_ms": _mean_stat(timing, "backend_total_ms"),
                "end_to_end_mean_ms": _mean_stat(timing, "end_to_end_ms"),
                "median_ms": end_to_end["median_ms"],
                "p95_ms": end_to_end["p95_ms"],
                "min_ms": end_to_end["min_ms"],
                "max_ms": end_to_end["max_ms"],
                "throughput_qps": end_to_end["fps"],
                "fallback_count": fallback_count,
                "error_count": error_count + int(factory.name in backend_errors),
                "mismatch_count": mismatch_count,
                "worker_pids": worker_pids,
                "worker_pid_reused": worker_pid_reused,
                "result": result,
            }
        )
    return rows


def write_reports(
    output_dir: Path,
    run: BenchmarkRun,
    canonical: dict[tuple[str, str], list[Detection]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(run.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_dict_csv(output_dir / "summary.csv", run.backend_rows)
    _write_dict_csv(
        output_dir / "per_image_results.csv",
        [_image_csv_row(row) for row in run.image_rows],
    )
    _write_dict_csv(
        output_dir / "detection_comparisons.csv",
        [asdict(row) for row in run.detection_rows],
        fieldnames=list(DetectionComparison.__dataclass_fields__),
    )
    _write_dict_csv(
        output_dir / "timing_samples.csv",
        [asdict(row) for row in run.timing_rows],
        fieldnames=list(TimingRecord.__dataclass_fields__),
    )
    (output_dir / "report.md").write_text(
        build_markdown_report(run),
        encoding="utf-8",
    )
    write_mismatch_cases(output_dir, run.comparisons, canonical)


def build_markdown_report(run: BenchmarkRun) -> str:
    config = run.summary["configuration"]
    lines = [
        "# Backend Comparison Report",
        "",
        "## Execution Environment",
        "",
        f"- Python: `{run.summary['environment']['python_version']}`",
        f"- Platform: `{run.summary['environment']['platform']}`",
        "",
        "## Models And Engines",
        "",
        f"- PyTorch: `{config['pytorch_model']}`",
        f"- ONNX: `{config['onnx_model']}`",
        f"- TensorRT FP32: `{config['tensorrt_fp32_engine']}`",
        f"- TensorRT FP16: `{config['tensorrt_fp16_engine']}`",
        "",
        "## Common Conditions",
        "",
        f"- Images: `{config['images']}` ({config['image_count']})",
        f"- imgsz/conf/iou/match_iou: `{config['imgsz']}` / `{config['conf']}` / `{config['iou']}` / `{config['match_iou']}`",
        f"- Warmup/repeat: `{config['warmup']}` / `{config['repeat']}`",
        f"- Device/provider: `{config['device']}` / `{config['provider']}`",
        f"- Warmup policy: {config['warmup_policy']}",
        "",
        "Backend stage timings are backend-reported values. End-to-end is the Python wall-clock duration. TensorRT startup is reported separately and excluded from steady-state statistics.",
        "",
        "## Backend Summary",
        "",
        "| Backend | Provider | Precision | Mismatch | Inference mean | End-to-end mean | p95 | Result |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in run.backend_rows:
        lines.append(
            f"| {row['backend']} | {row['provider'] or 'N/A'} | {row['precision']} | "
            f"{row['mismatch_count']} | {_format_ms(row['inference_mean_ms'])} | "
            f"{_format_ms(row['end_to_end_mean_ms'])} | {_format_ms(row['p95_ms'])} | "
            f"{row['result']} |"
        )
    lines.extend(
        [
            "",
            "## Accuracy Summary",
            "",
            "| Reference | Target | Images | Mismatch images | FP | FN | Class mismatch | Result |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    grouped: dict[tuple[str, str], list[ComparisonSummary]] = {}
    for item in run.comparisons:
        grouped.setdefault(
            (item.reference_backend, item.target_backend), []
        ).append(item)
    for pair, items in grouped.items():
        lines.append(
            f"| {pair[0]} | {pair[1]} | {len(items)} | "
            f"{sum(item.mismatch_count > 0 for item in items)} | "
            f"{sum(item.false_positive for item in items)} | "
            f"{sum(item.false_negative for item in items)} | "
            f"{sum(item.class_mismatch for item in items)} | "
            f"{_overall_status([item.status for item in items])} |"
        )
    lines.extend(
        [
            "",
            "## TensorRT Worker",
            "",
        ]
    )
    for row in run.backend_rows:
        if row["backend"].startswith("tensorrt"):
            lines.append(
                f"- {row['backend']}: startup `{_format_ms(row['startup_ms'])}`, "
                f"first request `{_format_ms(row['first_request_ms'])}`, "
                f"PID reused `{row['worker_pid_reused']}`, fallbacks `{row['fallback_count']}`."
            )
    errors = run.summary.get("backend_errors", {})
    lines.extend(["", "## Fallbacks And Errors", ""])
    if errors:
        lines.extend(f"- {backend}: {message}" for backend, message in errors.items())
    else:
        lines.append("- No backend initialization errors were recorded.")
    lines.extend(
        [
            "",
            "## Final Conclusion",
            "",
            f"Final status: **{run.summary['final_status']}**",
            "",
        ]
    )
    return "\n".join(lines)


def write_mismatch_cases(
    output_dir: Path,
    comparisons: list[ComparisonSummary],
    canonical: dict[tuple[str, str], list[Detection]],
) -> None:
    mismatch_root = output_dir / "mismatch_cases"
    if mismatch_root.exists():
        shutil.rmtree(mismatch_root)
    mismatch_images = {
        item.image for item in comparisons if item.status != "PASS"
    }
    if not mismatch_images:
        return
    for image_name in sorted(mismatch_images):
        case_dir = mismatch_root / Path(image_name).stem
        case_dir.mkdir(parents=True, exist_ok=True)
        for backend in BACKEND_ORDER:
            detections = next(
                (
                    values
                    for (name, image_path), values in canonical.items()
                    if name == backend and Path(image_path).name == image_name
                ),
                None,
            )
            if detections is not None:
                (case_dir / f"{backend}.json").write_text(
                    json.dumps(
                        {
                            "backend": backend,
                            "image": image_name,
                            "detections": [
                                detection_to_dict(item) for item in detections
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )


def _timing_record(
    image_path: Path,
    backend: str,
    run_index: int,
    is_warmup: bool,
    values: dict[str, float | None],
    host_ms: float,
    worker_pid: int | None = None,
    fallback_used: bool = False,
) -> TimingRecord:
    return TimingRecord(
        image=image_path.name,
        backend=backend,
        run_index=run_index,
        is_warmup=is_warmup,
        preprocess_ms=_optional_float(values.get("preprocess_ms")),
        inference_ms=_optional_float(values.get("inference_ms")),
        postprocess_ms=_optional_float(values.get("postprocess_ms")),
        backend_total_ms=_optional_float(values.get("total_ms")),
        host_roundtrip_ms=float(host_ms),
        end_to_end_ms=float(host_ms),
        worker_pid=worker_pid,
        fallback_used=fallback_used,
    )


def _runner_metadata(runner: BackendRunner) -> dict[str, Any]:
    if not isinstance(runner, TensorRtBackendRunner):
        return {}
    worker = runner.adapter._persistent_worker
    pid = worker.pid if worker is not None else None
    return {
        "worker_pids": [pid] if pid is not None else [],
        "worker_pid_reused": pid is not None,
    }


def _pair_class_mismatches(
    reference_only: list[DetectionMatch],
    target_only: list[DetectionMatch],
    match_iou: float,
) -> list[tuple[DetectionMatch, DetectionMatch, float]]:
    from service.onnx_detector import bbox_iou

    candidates: list[tuple[float, DetectionMatch, DetectionMatch]] = []
    for left in reference_only:
        for right in target_only:
            if left.pt is None or right.onnx is None:
                continue
            overlap = bbox_iou(
                [left.pt.x1, left.pt.y1, left.pt.x2, left.pt.y2],
                [right.onnx.x1, right.onnx.y1, right.onnx.x2, right.onnx.y2],
            )
            if overlap >= match_iou:
                candidates.append((overlap, left, right))
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[tuple[DetectionMatch, DetectionMatch, float]] = []
    for overlap, left, right in sorted(candidates, key=lambda item: item[0], reverse=True):
        if id(left) in used_left or id(right) in used_right:
            continue
        used_left.add(id(left))
        used_right.add(id(right))
        pairs.append((left, right, overlap))
    return pairs


def _detection_comparison(
    image: str,
    reference_backend: str,
    target_backend: str,
    reference_index: int | None,
    target_index: int | None,
    reference: Detection | None,
    target: Detection | None,
    confidence_delta: float | None,
    bbox_iou: float | None,
    status: str,
    reason: str,
) -> DetectionComparison:
    return DetectionComparison(
        image=image,
        reference_backend=reference_backend,
        target_backend=target_backend,
        reference_index=reference_index,
        target_index=target_index,
        reference_class=reference.class_name if reference is not None else "",
        target_class=target.class_name if target is not None else "",
        reference_confidence=(
            reference.confidence if reference is not None else None
        ),
        target_confidence=target.confidence if target is not None else None,
        confidence_delta=confidence_delta,
        bbox_iou=bbox_iou,
        match_status=status,
        reason=reason,
    )


def _comparison_has_mismatch(summary: ComparisonSummary) -> bool:
    return summary.status == "FAIL"


def _image_csv_row(row: ImageBackendResult) -> dict[str, Any]:
    data = asdict(row)
    data["class_names"] = ",".join(row.class_names)
    data.pop("detections", None)
    data.pop("stability_mismatch_count", None)
    return data


def _write_dict_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding=CSV_ENCODING) as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _overall_status(statuses: list[str]) -> str:
    if "FAIL" in statuses or "ERROR" in statuses:
        return "FAIL"
    if "WARNING" in statuses:
        return "WARNING"
    if statuses == ["BASELINE"]:
        return "BASELINE"
    return "PASS"


def _mean_stat(
    summary: dict[str, dict[str, float | None]],
    field_name: str,
) -> float | None:
    return summary.get(field_name, {}).get("mean_ms")


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _format_ms(value: object) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{float(value):.3f} ms"


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run = run_benchmark(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Final status: {run.summary['final_status']}")
    print(f"Reports: {args.output}")
    return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
