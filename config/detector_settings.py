from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSettings

from config.settings import PROJECT_ROOT


DETECTOR_BACKEND_LABELS = {
    "pytorch": "PyTorch",
    "onnx": "ONNX Runtime",
    "tensorrt": "TensorRT",
}

DETECTOR_BACKEND_VALUES = {label: value for value, label in DETECTOR_BACKEND_LABELS.items()}

ENGINE_LABELS = {
    "fp16": "FP16",
    "fp32": "FP32",
}

ENGINE_LABEL_VALUES = {label: value for value, label in ENGINE_LABELS.items()}


@dataclass(frozen=True, slots=True)
class DetectorSettings:
    detector_backend: str = "pytorch"
    tensorrt_executable_path: str = "cpp_inference/build_gpu/Release/pcb_onnx_infer.exe"
    tensorrt_engine_path: str = "benchmarks/tensorrt/best_fp16.engine"
    tensorrt_engine_label: str = "fp16"
    tensorrt_metadata_path: str = "models/model_metadata.json"
    tensorrt_device_id: int = 0
    tensorrt_timeout_seconds: float = 120.0
    keep_tensorrt_outputs: bool = False
    tensorrt_use_persistent_worker: bool = True
    tensorrt_fallback_to_oneshot: bool = True
    tensorrt_worker_startup_timeout_seconds: float = 120.0


def qsettings() -> QSettings:
    return QSettings("V_VLM", "PCB Vision Inspection")


def load_detector_settings(settings: QSettings | None = None) -> DetectorSettings:
    store = settings or qsettings()
    return DetectorSettings(
        detector_backend=_backend_value(str(store.value("detector_backend", "pytorch"))),
        tensorrt_executable_path=str(
            store.value(
                "tensorrt_executable_path",
                "cpp_inference/build_gpu/Release/pcb_onnx_infer.exe",
            )
        ),
        tensorrt_engine_path=str(
            store.value("tensorrt_engine_path", "benchmarks/tensorrt/best_fp16.engine")
        ),
        tensorrt_engine_label=_engine_label_value(str(store.value("tensorrt_engine_label", "fp16"))),
        tensorrt_metadata_path=str(store.value("tensorrt_metadata_path", "models/model_metadata.json")),
        tensorrt_device_id=max(0, _int_value(store.value("tensorrt_device_id", 0), default=0)),
        tensorrt_timeout_seconds=max(
            1.0,
            _float_value(store.value("tensorrt_timeout_seconds", 120.0), default=120.0),
        ),
        keep_tensorrt_outputs=_bool_value(store.value("keep_tensorrt_outputs", False)),
        tensorrt_use_persistent_worker=_bool_value(
            store.value("tensorrt_use_persistent_worker", True)
        ),
        tensorrt_fallback_to_oneshot=_bool_value(
            store.value("tensorrt_fallback_to_oneshot", True)
        ),
        tensorrt_worker_startup_timeout_seconds=max(
            1.0,
            _float_value(
                store.value("tensorrt_worker_startup_timeout_seconds", 120.0),
                default=120.0,
            ),
        ),
    )


def save_detector_settings(detector_settings: DetectorSettings, settings: QSettings | None = None) -> None:
    store = settings or qsettings()
    store.setValue("detector_backend", detector_settings.detector_backend)
    store.setValue("tensorrt_executable_path", detector_settings.tensorrt_executable_path)
    store.setValue("tensorrt_engine_path", detector_settings.tensorrt_engine_path)
    store.setValue("tensorrt_engine_label", detector_settings.tensorrt_engine_label)
    store.setValue("tensorrt_metadata_path", detector_settings.tensorrt_metadata_path)
    store.setValue("tensorrt_device_id", detector_settings.tensorrt_device_id)
    store.setValue("tensorrt_timeout_seconds", detector_settings.tensorrt_timeout_seconds)
    store.setValue("keep_tensorrt_outputs", detector_settings.keep_tensorrt_outputs)
    store.setValue(
        "tensorrt_use_persistent_worker",
        detector_settings.tensorrt_use_persistent_worker,
    )
    store.setValue(
        "tensorrt_fallback_to_oneshot",
        detector_settings.tensorrt_fallback_to_oneshot,
    )
    store.setValue(
        "tensorrt_worker_startup_timeout_seconds",
        detector_settings.tensorrt_worker_startup_timeout_seconds,
    )
    store.sync()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_tensorrt_settings(detector_settings: DetectorSettings) -> None:
    if detector_settings.detector_backend != "tensorrt":
        return

    executable = resolve_project_path(detector_settings.tensorrt_executable_path)
    engine = resolve_project_path(detector_settings.tensorrt_engine_path)
    metadata = resolve_project_path(detector_settings.tensorrt_metadata_path)

    if not executable.is_file():
        raise ValueError(f"TensorRT 실행 파일을 찾을 수 없습니다: {executable}")
    if executable.suffix.lower() != ".exe":
        raise ValueError("TensorRT 실행 파일은 .exe 파일이어야 합니다.")
    if not engine.is_file():
        raise ValueError(f"TensorRT 엔진 파일을 찾을 수 없습니다: {engine}")
    if engine.suffix.lower() not in {".engine", ".plan"}:
        raise ValueError("TensorRT 엔진 파일은 .engine 또는 .plan 파일이어야 합니다.")
    if not metadata.is_file():
        raise ValueError(f"모델 metadata 파일을 찾을 수 없습니다: {metadata}")
    if metadata.suffix.lower() != ".json":
        raise ValueError("모델 metadata 파일은 .json 파일이어야 합니다.")
    if detector_settings.tensorrt_device_id < 0:
        raise ValueError("CUDA Device ID는 0 이상이어야 합니다.")
    if detector_settings.tensorrt_engine_label not in ENGINE_LABELS:
        raise ValueError("TensorRT 엔진 정밀도는 fp16 또는 fp32여야 합니다.")


def backend_display_text(detector_settings: DetectorSettings) -> str:
    if detector_settings.detector_backend == "tensorrt":
        precision = ENGINE_LABELS.get(detector_settings.tensorrt_engine_label, detector_settings.tensorrt_engine_label.upper())
        return f"TensorRT {precision}"
    return DETECTOR_BACKEND_LABELS.get(detector_settings.detector_backend, detector_settings.detector_backend)


def backend_detail_text(detector_settings: DetectorSettings) -> str:
    if detector_settings.detector_backend == "tensorrt":
        engine = resolve_project_path(detector_settings.tensorrt_engine_path)
        return (
            f"Precision: {ENGINE_LABELS.get(detector_settings.tensorrt_engine_label, detector_settings.tensorrt_engine_label.upper())} | "
            f"CUDA {detector_settings.tensorrt_device_id} | Engine: {engine.name}"
        )
    if detector_settings.detector_backend == "onnx":
        return "ONNX Runtime detector"
    return "Default PyTorch detector"


def _backend_value(value: str) -> str:
    lowered = value.lower()
    return lowered if lowered in DETECTOR_BACKEND_LABELS else "pytorch"


def _engine_label_value(value: str) -> str:
    lowered = value.lower()
    return lowered if lowered in ENGINE_LABELS else "fp16"


def _int_value(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)
