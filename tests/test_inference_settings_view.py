from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox

from config.detector_settings import DetectorSettings
from view.inference_settings_view import InferenceSettingsView
from view.main_window import MainWindow


def _app(monkeypatch: pytest.MonkeyPatch) -> QApplication:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication(sys.argv)


def _tensorrt_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    executable = tmp_path / "pcb_onnx_infer.exe"
    engine = tmp_path / "best_fp16.engine"
    metadata = tmp_path / "model_metadata.json"
    executable.write_text("exe", encoding="utf-8")
    engine.write_bytes(b"engine")
    metadata.write_text("{}", encoding="utf-8")
    return executable, engine, metadata


class FakeInferenceSettingsViewModel:
    def __init__(self, settings: DetectorSettings | None = None) -> None:
        self.settings = settings or DetectorSettings()
        self.saved: DetectorSettings | None = None
        self.validated: DetectorSettings | None = None

    def get_detector_settings(self) -> DetectorSettings:
        return self.settings

    def save_detector_settings(self, settings: DetectorSettings) -> None:
        self.saved = settings
        self.settings = settings

    def validate_detector_settings(self, settings: DetectorSettings) -> None:
        self.validated = settings

    def backend_display_text(self) -> str:
        if self.settings.detector_backend == "tensorrt":
            return f"TensorRT {self.settings.tensorrt_engine_label.upper()}"
        if self.settings.detector_backend == "onnx":
            return "ONNX Runtime"
        return "PyTorch"

    def backend_detail_text(self) -> str:
        return f"device={self.settings.tensorrt_device_id}"


def test_inference_settings_view_toggles_tensorrt_fields(monkeypatch) -> None:
    _app(monkeypatch)
    view = InferenceSettingsView(FakeInferenceSettingsViewModel())

    assert view.executable_edit.isEnabled() is False

    view.backend_combo.setCurrentIndex(view.backend_combo.findData("tensorrt"))

    assert view.executable_edit.isEnabled() is True
    assert view.engine_edit.isEnabled() is True
    view.close()


def test_inference_settings_view_saves_and_refreshes(monkeypatch, tmp_path) -> None:
    _app(monkeypatch)
    executable, engine, metadata = _tensorrt_files(tmp_path)
    viewmodel = FakeInferenceSettingsViewModel()
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    view = InferenceSettingsView(viewmodel)

    view.backend_combo.setCurrentIndex(view.backend_combo.findData("tensorrt"))
    view.executable_edit.setText(str(executable))
    view.engine_edit.setText(str(engine))
    view.metadata_edit.setText(str(metadata))
    view.device_spin.setValue(1)
    view._save()

    assert viewmodel.saved is not None
    assert viewmodel.saved.detector_backend == "tensorrt"
    assert viewmodel.saved.tensorrt_engine_path == str(engine)
    assert viewmodel.saved.tensorrt_device_id == 1
    assert view.current_backend_value.text() == "TensorRT FP16"
    view.close()


def test_inference_settings_view_validates_without_saving(monkeypatch, tmp_path) -> None:
    _app(monkeypatch)
    executable, engine, metadata = _tensorrt_files(tmp_path)
    viewmodel = FakeInferenceSettingsViewModel()
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    view = InferenceSettingsView(viewmodel)

    view.backend_combo.setCurrentIndex(view.backend_combo.findData("tensorrt"))
    view.executable_edit.setText(str(executable))
    view.engine_edit.setText(str(engine))
    view.metadata_edit.setText(str(metadata))
    assert view._validate() is True

    assert viewmodel.validated is not None
    assert viewmodel.saved is None
    view.close()


def test_main_window_registers_inference_settings_page(monkeypatch) -> None:
    _app(monkeypatch)
    window = MainWindow()

    assert window.stack.count() == 5
    assert window.inference_settings_tab.text() == "추론 설정"

    window.inference_settings_tab.click()

    assert window.stack.currentIndex() == 4
    window.close()
