from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config.detector_settings import DETECTOR_BACKEND_LABELS, ENGINE_LABELS, DetectorSettings
from viewmodel.inference_settings_viewmodel import InferenceSettingsViewModel


class InferenceSettingsView(QWidget):
    def __init__(self, viewmodel: InferenceSettingsViewModel | None = None) -> None:
        super().__init__()
        self.viewmodel = viewmodel or InferenceSettingsViewModel()

        self.current_backend_value = QLabel("-")
        self.current_backend_value.setObjectName("StatusValue")
        self.current_backend_detail = QLabel("")
        self.current_backend_detail.setObjectName("StatusDetail")
        self.current_backend_detail.setWordWrap(True)

        settings = self.viewmodel.get_detector_settings()
        self.backend_combo = QComboBox()
        for value, label in DETECTOR_BACKEND_LABELS.items():
            self.backend_combo.addItem(label, value)
        self.backend_combo.setCurrentIndex(max(0, self.backend_combo.findData(settings.detector_backend)))

        self.executable_edit = QLineEdit(settings.tensorrt_executable_path)
        self.engine_edit = QLineEdit(settings.tensorrt_engine_path)
        self.metadata_edit = QLineEdit(settings.tensorrt_metadata_path)

        self.engine_label_combo = QComboBox()
        for value, label in ENGINE_LABELS.items():
            self.engine_label_combo.addItem(label, value)
        self.engine_label_combo.setCurrentIndex(max(0, self.engine_label_combo.findData(settings.tensorrt_engine_label)))

        self.device_spin = QSpinBox()
        self.device_spin.setRange(0, 64)
        self.device_spin.setValue(settings.tensorrt_device_id)

        self.save_button = QPushButton("설정 저장")
        self.validate_button = QPushButton("설정 검증")
        self.save_button.setObjectName("PrimaryButton")
        self.validate_button.setObjectName("SecondaryButton")
        self.save_button.clicked.connect(self._save)
        self.validate_button.clicked.connect(self._validate)
        self.backend_combo.currentIndexChanged.connect(self._sync_enabled_state)

        self._build_layout()
        self._sync_enabled_state()
        self.refresh()
        self.setStyleSheet(_inference_settings_stylesheet())

    def refresh(self) -> None:
        settings = self.viewmodel.get_detector_settings()
        self.current_backend_value.setText(self.viewmodel.backend_display_text())
        self.current_backend_detail.setText(self.viewmodel.backend_detail_text())
        self.backend_combo.setCurrentIndex(max(0, self.backend_combo.findData(settings.detector_backend)))
        self.executable_edit.setText(settings.tensorrt_executable_path)
        self.engine_edit.setText(settings.tensorrt_engine_path)
        self.engine_label_combo.setCurrentIndex(max(0, self.engine_label_combo.findData(settings.tensorrt_engine_label)))
        self.metadata_edit.setText(settings.tensorrt_metadata_path)
        self.device_spin.setValue(settings.tensorrt_device_id)
        self._sync_enabled_state()

    def _build_layout(self) -> None:
        current_card = QFrame()
        current_card.setObjectName("Panel")
        current_layout = QVBoxLayout(current_card)
        current_layout.setContentsMargins(18, 16, 18, 16)
        current_layout.setSpacing(8)
        title = QLabel("현재 추론 Backend")
        title.setObjectName("StatusTitle")
        current_layout.addWidget(title)
        current_layout.addWidget(self.current_backend_value)
        current_layout.addWidget(self.current_backend_detail)

        settings_card = QFrame()
        settings_card.setObjectName("Panel")
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(18, 16, 18, 16)
        settings_layout.setSpacing(10)
        settings_title = QLabel("추론 설정")
        settings_title.setObjectName("StatusTitle")
        settings_layout.addWidget(settings_title)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.addRow("추론 Backend", self.backend_combo)
        form.addRow("TensorRT 실행 파일", self._path_row(self.executable_edit, "Executable (*.exe)"))
        form.addRow("TensorRT 엔진", self._path_row(self.engine_edit, "TensorRT Engine (*.engine *.plan)"))
        form.addRow("엔진 정밀도", self.engine_label_combo)
        form.addRow("모델 Metadata", self._path_row(self.metadata_edit, "JSON (*.json)"))
        form.addRow("CUDA Device ID", self.device_spin)
        settings_layout.addLayout(form)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.validate_button)
        button_row.addWidget(self.save_button)
        settings_layout.addLayout(button_row)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 8, 24, 16)
        layout.setSpacing(16)
        layout.addWidget(current_card)
        layout.addWidget(settings_card)
        layout.addStretch(1)

    def _path_row(self, edit: QLineEdit, file_filter: str) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        browse_button = QPushButton("찾기")
        browse_button.clicked.connect(lambda: self._browse_file(edit, file_filter))
        layout.addWidget(edit, 1)
        layout.addWidget(browse_button)
        return panel

    def _browse_file(self, edit: QLineEdit, file_filter: str) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "파일 선택", edit.text(), file_filter)
        if selected:
            edit.setText(selected)

    def _sync_enabled_state(self, *_args: object) -> None:
        enabled = self.backend_combo.currentData() == "tensorrt"
        for widget in (
            self.executable_edit,
            self.engine_edit,
            self.engine_label_combo,
            self.metadata_edit,
            self.device_spin,
        ):
            widget.setEnabled(enabled)

    def _settings_from_inputs(self) -> DetectorSettings:
        current = self.viewmodel.get_detector_settings()
        return DetectorSettings(
            detector_backend=str(self.backend_combo.currentData() or "pytorch"),
            tensorrt_executable_path=self.executable_edit.text().strip(),
            tensorrt_engine_path=self.engine_edit.text().strip(),
            tensorrt_engine_label=str(self.engine_label_combo.currentData() or "fp16"),
            tensorrt_metadata_path=self.metadata_edit.text().strip(),
            tensorrt_device_id=self.device_spin.value(),
            tensorrt_timeout_seconds=current.tensorrt_timeout_seconds,
            keep_tensorrt_outputs=current.keep_tensorrt_outputs,
        )

    def _validate(self) -> bool:
        try:
            self.viewmodel.validate_detector_settings(self._settings_from_inputs())
        except ValueError as exc:
            QMessageBox.warning(self, "TensorRT 설정 오류", str(exc))
            return False
        QMessageBox.information(self, "설정 검증", "추론 Backend 설정을 사용할 수 있습니다.")
        return True

    def _save(self) -> None:
        try:
            self.viewmodel.save_detector_settings(self._settings_from_inputs())
        except ValueError as exc:
            QMessageBox.warning(self, "TensorRT 설정 오류", str(exc))
            return
        QMessageBox.information(self, "설정 저장", "추론 Backend 설정을 저장했습니다. 다음 검사부터 적용됩니다.")
        self.refresh()


def _inference_settings_stylesheet() -> str:
    return """
    QFrame#Panel {
        background: #ffffff;
        border: 1px solid #d6dde8;
        border-radius: 6px;
    }
    QLabel#StatusTitle {
        color: #17202a;
        font-size: 14px;
        font-weight: 800;
    }
    QLabel#StatusValue {
        color: #263241;
        font-size: 22px;
        font-weight: 900;
    }
    QLabel#StatusDetail {
        color: #667085;
        font-size: 12px;
    }
    QPushButton#PrimaryButton, QPushButton#SecondaryButton {
        background: #ffffff;
        color: #263241;
        border: 1px solid #ccd4df;
        border-radius: 6px;
        padding: 8px 14px;
        font-weight: 700;
    }
    QPushButton#PrimaryButton {
        background: #e8f1ff;
        color: #1250b5;
        border-color: #9dbcf5;
    }
    QPushButton#PrimaryButton:hover, QPushButton#SecondaryButton:hover {
        background: #eef4ff;
        border-color: #8fb5ff;
    }
    QLineEdit, QComboBox, QSpinBox {
        background: #ffffff;
        border: 1px solid #ccd4df;
        border-radius: 6px;
        padding: 6px 8px;
        min-height: 22px;
    }
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {
        background: #eef1f5;
        color: #8a94a3;
    }
    """
