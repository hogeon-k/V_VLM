from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings

from config.detector_settings import (
    DetectorSettings,
    backend_display_text,
    load_detector_settings,
    save_detector_settings,
    validate_tensorrt_settings,
)
from service import detector_backend_factory
from service.detector_backend_factory import create_yolo_service_from_settings
from service.onnx_detector import OnnxDetector
from yolo.detector import YoloDetector


def _qsettings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def _tensorrt_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    executable = tmp_path / "pcb_onnx_infer.exe"
    engine = tmp_path / "best_fp16.engine"
    metadata = tmp_path / "model_metadata.json"
    executable.write_text("exe", encoding="utf-8")
    engine.write_bytes(b"engine")
    metadata.write_text("{}", encoding="utf-8")
    return executable, engine, metadata


def test_detector_settings_defaults_to_pytorch_when_keys_are_missing(tmp_path) -> None:
    loaded = load_detector_settings(_qsettings(tmp_path))

    assert loaded.detector_backend == "pytorch"
    assert loaded.tensorrt_engine_label == "fp16"
    assert loaded.tensorrt_device_id == 0
    assert loaded.keep_tensorrt_outputs is False
    assert backend_display_text(loaded) == "PyTorch"


def test_detector_settings_save_and_restore_tensorrt_values(tmp_path) -> None:
    store = _qsettings(tmp_path)
    original = DetectorSettings(
        detector_backend="tensorrt",
        tensorrt_executable_path="bin/infer.exe",
        tensorrt_engine_path="engines/best.plan",
        tensorrt_engine_label="fp32",
        tensorrt_metadata_path="models/meta.json",
        tensorrt_device_id=2,
        keep_tensorrt_outputs=True,
        tensorrt_use_persistent_worker=False,
        tensorrt_fallback_to_oneshot=False,
        tensorrt_worker_startup_timeout_seconds=45.0,
    )

    save_detector_settings(original, store)
    loaded = load_detector_settings(store)

    assert loaded == original
    assert backend_display_text(loaded) == "TensorRT FP32"


def test_tensorrt_validation_reports_missing_executable(tmp_path) -> None:
    _, engine, metadata = _tensorrt_files(tmp_path)
    settings = DetectorSettings(
        detector_backend="tensorrt",
        tensorrt_executable_path=str(tmp_path / "missing.exe"),
        tensorrt_engine_path=str(engine),
        tensorrt_metadata_path=str(metadata),
    )

    with pytest.raises(ValueError, match="TensorRT 실행 파일"):
        validate_tensorrt_settings(settings)


def test_tensorrt_validation_rejects_bad_engine_extension(tmp_path) -> None:
    executable, _, metadata = _tensorrt_files(tmp_path)
    bad_engine = tmp_path / "best.txt"
    bad_engine.write_text("not engine", encoding="utf-8")
    settings = DetectorSettings(
        detector_backend="tensorrt",
        tensorrt_executable_path=str(executable),
        tensorrt_engine_path=str(bad_engine),
        tensorrt_metadata_path=str(metadata),
    )

    with pytest.raises(ValueError, match=".engine"):
        validate_tensorrt_settings(settings)


def test_backend_factory_keeps_pytorch_default() -> None:
    service = create_yolo_service_from_settings(DetectorSettings())

    assert isinstance(service.detector, YoloDetector)


def test_backend_factory_selects_onnx() -> None:
    service = create_yolo_service_from_settings(DetectorSettings(detector_backend="onnx"))

    assert isinstance(service.detector, OnnxDetector)


def test_backend_factory_passes_tensorrt_arguments(monkeypatch, tmp_path) -> None:
    executable, engine, metadata = _tensorrt_files(tmp_path)
    captured: dict[str, object] = {}

    class FakeTensorRtDetectorAdapter:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(detector_backend_factory, "TensorRtDetectorAdapter", FakeTensorRtDetectorAdapter)

    service = create_yolo_service_from_settings(
        DetectorSettings(
            detector_backend="tensorrt",
            tensorrt_executable_path=str(executable),
            tensorrt_engine_path=str(engine),
            tensorrt_engine_label="fp16",
            tensorrt_metadata_path=str(metadata),
            tensorrt_device_id=3,
        )
    )

    assert isinstance(service.detector, FakeTensorRtDetectorAdapter)
    assert captured["executable_path"] == executable
    assert captured["engine_path"] == engine
    assert captured["metadata_path"] == metadata
    assert captured["device_id"] == 3
    assert captured["engine_label"] == "fp16"
    assert captured["use_persistent_worker"] is True
    assert captured["fallback_to_oneshot"] is True
    assert captured["worker_startup_timeout_seconds"] == 120.0
