from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from viewmodel.inspection_viewmodel import InspectionViewModel


class InspectionView(QWidget):
    def __init__(self, viewmodel: InspectionViewModel | None = None) -> None:
        super().__init__()
        self.viewmodel = viewmodel or InspectionViewModel()
        self._paused = False

        self.folder_label = QLabel("No folder selected")
        self.progress_label = QLabel("Idle")
        self.current_image_label = _image_label("Current Image")
        self.result_image_label = _image_label("Result Image")
        self.status_label = QLabel("PENDING")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 28px; font-weight: 700;")
        self.description = QTextEdit()
        self.description.setReadOnly(True)

        choose_button = QPushButton("Select Folder")
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        choose_button.clicked.connect(self._choose_folder)
        self.start_button.clicked.connect(self._start)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self.viewmodel.stop)

        top = QHBoxLayout()
        top.addWidget(choose_button)
        top.addWidget(self.folder_label, 1)
        top.addWidget(self.start_button)
        top.addWidget(self.pause_button)
        top.addWidget(self.stop_button)

        images = QGridLayout()
        images.addWidget(self.current_image_label, 0, 0)
        images.addWidget(self.result_image_label, 0, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.progress_label)
        layout.addLayout(images, 1)
        layout.addWidget(self.status_label)
        layout.addWidget(self.description, 1)

        self.viewmodel.started.connect(self._on_started)
        self.viewmodel.image_started.connect(self._on_image_started)
        self.viewmodel.result_ready.connect(self._on_result_ready)
        self.viewmodel.error.connect(self._show_error)
        self.viewmodel.finished.connect(self._on_finished)

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        try:
            images = self.viewmodel.select_folder(folder)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.folder_label.setText(f"{folder} ({len(images)} images)")
        self.progress_label.setText("Ready")

    def _start(self) -> None:
        try:
            self.viewmodel.start_auto_inspection()
        except Exception as exc:
            self._show_error(str(exc))

    def _toggle_pause(self) -> None:
        if self._paused:
            self.viewmodel.resume()
            self.pause_button.setText("Pause")
            self._paused = False
        else:
            self.viewmodel.pause()
            self.pause_button.setText("Resume")
            self._paused = True

    def _on_started(self, total: int) -> None:
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.progress_label.setText(f"Running: 0 / {total}")

    def _on_image_started(self, image_path: str, index: int, total: int) -> None:
        self.progress_label.setText(f"Running: {index} / {total} - {Path(image_path).name}")
        _set_pixmap(self.current_image_label, Path(image_path))
        self.status_label.setText("RUNNING")
        self.description.clear()

    def _on_result_ready(self, result: object) -> None:
        self.status_label.setText(getattr(result, "status", "UNKNOWN"))
        result_path = getattr(result, "result_image_path", None)
        if result_path:
            _set_pixmap(self.result_image_label, Path(result_path))
        description = getattr(result, "vlm_description", None) or "OK - no defect detected."
        self.description.setPlainText(description)

    def _on_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self._paused = False
        self.progress_label.setText("Finished")

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Inspection", message)


def _image_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignCenter)
    label.setMinimumHeight(260)
    label.setStyleSheet("border: 1px solid #c8ccd2; background: #f7f8fa;")
    return label


def _set_pixmap(label: QLabel, path: Path) -> None:
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        label.setText(f"Failed to load image:\n{path}")
        return
    label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
