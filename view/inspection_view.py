from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from viewmodel.inspection_viewmodel import InspectionViewModel


class InspectionView(QWidget):
    def __init__(self, viewmodel: InspectionViewModel | None = None) -> None:
        super().__init__()
        self.viewmodel = viewmodel or InspectionViewModel()
        self._paused = False
        self._stop_requested = False
        self._total_count = 0
        self._current_index = 0

        self.folder_label = QLabel("검사 폴더가 선택되지 않았습니다.")
        self.folder_label.setObjectName("MutedText")

        self.choose_button = QPushButton("이미지 폴더 선택")
        self.start_button = QPushButton("자동 검사 시작")
        self.pause_button = QPushButton("일시정지")
        self.stop_button = QPushButton("정지")
        for button in (self.choose_button, self.start_button, self.pause_button, self.stop_button):
            button.setMinimumHeight(34)
            button.setMinimumWidth(124)

        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        self.current_image_panel = ImagePanel("현재 검사 이미지", "검사할 이미지가 없습니다.")
        self.result_image_panel = ImagePanel("결과 이미지 (Bounding Box)", "결과 이미지가 없습니다.")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("검사 진행률: 0 / 0")
        self.state_label = QLabel("현재 상태: 대기")
        self.progress_label.setObjectName("MetricValue")
        self.state_label.setObjectName("MetricValue")
        self.progress_label.setMaximumHeight(34)
        self.state_label.setMaximumHeight(34)

        self.judgment_label = QLabel("미판정")
        self.defect_type_label = QLabel("-")
        self.confidence_label = QLabel("-")
        self.judgment_label.setObjectName("JudgmentNeutral")
        self.judgment_label.setAlignment(Qt.AlignCenter)
        self.judgment_label.setMaximumHeight(50)

        self.description = QPlainTextEdit()
        self.description.setReadOnly(True)
        self.description.setPlainText("VLM 설명은 아직 생성되지 않았습니다.\n이력 화면에서 VLM 설명 생성을 실행할 수 있습니다.")
        self.description.setMinimumHeight(96)

        self.choose_button.clicked.connect(self._choose_folder)
        self.start_button.clicked.connect(self._start_or_resume)
        self.pause_button.clicked.connect(self._pause)
        self.stop_button.clicked.connect(self._stop)

        self._build_layout()
        self._connect_viewmodel()
        self._apply_idle_state()
        self.setStyleSheet(_inspection_stylesheet())

    def _build_layout(self) -> None:
        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(self.choose_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.folder_label, 1)

        image_grid = QGridLayout()
        image_grid.setHorizontalSpacing(14)
        image_grid.setColumnStretch(0, 1)
        image_grid.setColumnStretch(1, 1)
        image_grid.addWidget(self.current_image_panel, 0, 0)
        image_grid.addWidget(self.result_image_panel, 0, 1)

        status_row = QGridLayout()
        status_row.setHorizontalSpacing(14)
        status_row.setVerticalSpacing(6)
        status_row.addWidget(_section_title("검사 진행률"), 0, 0)
        status_row.addWidget(_section_title("현재 상태"), 0, 1)
        status_row.addWidget(self.progress_label, 1, 0)
        status_row.addWidget(self.state_label, 1, 1)
        status_row.addWidget(self.progress_bar, 2, 0, 1, 2)
        status_row.setColumnStretch(0, 1)
        status_row.setColumnStretch(1, 1)

        result_grid = QGridLayout()
        result_grid.setHorizontalSpacing(14)
        result_grid.setVerticalSpacing(6)
        result_grid.setColumnStretch(0, 2)
        result_grid.setColumnStretch(1, 1)
        result_grid.setColumnStretch(2, 1)
        result_grid.addWidget(_section_title("판정 결과"), 0, 0)
        result_grid.addWidget(_section_title("불량 유형"), 0, 1)
        result_grid.addWidget(_section_title("신뢰도"), 0, 2)
        result_grid.addWidget(self.judgment_label, 1, 0)
        result_grid.addWidget(_value_panel(self.defect_type_label), 1, 1)
        result_grid.addWidget(_value_panel(self.confidence_label), 1, 2)

        description_header = _section_title("VLM 분석 결과")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 8, 24, 16)
        layout.setSpacing(10)
        layout.addLayout(controls)
        layout.addLayout(image_grid, 9)
        layout.addLayout(status_row)
        layout.addLayout(result_grid)
        layout.addWidget(description_header)
        layout.addWidget(self.description, 1)

    def _connect_viewmodel(self) -> None:
        self.viewmodel.started.connect(self._on_started)
        self.viewmodel.image_started.connect(self._on_image_started)
        self.viewmodel.result_ready.connect(self._on_result_ready)
        self.viewmodel.error.connect(self._show_error)
        self.viewmodel.finished.connect(self._on_finished)

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "검사 이미지 폴더 선택")
        if not folder:
            return
        try:
            images = self.viewmodel.select_folder(folder)
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._total_count = len(images)
        self._current_index = 0
        self.folder_label.setText(f"{folder} ({len(images)}개 이미지)")
        self.current_image_panel.set_image(images[0])
        self.result_image_panel.clear_image("결과 이미지가 없습니다.")
        self.progress_bar.setRange(0, max(1, len(images)))
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"검사 진행률: 0 / {len(images)}")
        self._set_status("대기")
        self._set_judgment(None, [], None)
        self.description.setPlainText("자동 검사 시작 버튼을 누르면 선택한 폴더의 이미지를 순서대로 검사합니다.")
        self._apply_idle_state()

    def _start_or_resume(self) -> None:
        if self._paused:
            self.viewmodel.resume()
            self._paused = False
            self._stop_requested = False
            self._apply_running_state()
            self._set_status("검사 중")
            return

        if not self.viewmodel.image_paths:
            self._show_error("먼저 검사 대상 이미지 폴더를 선택하세요.")
            return

        try:
            self._stop_requested = False
            self.viewmodel.start_auto_inspection()
        except Exception as exc:
            self._show_error(str(exc))

    def _pause(self) -> None:
        if not self.viewmodel.is_running():
            return
        self.viewmodel.pause()
        self._paused = True
        self._apply_paused_state()
        self._set_status("일시정지")

    def _stop(self) -> None:
        if self.viewmodel.is_running():
            self._stop_requested = True
            self.viewmodel.stop()
        self._paused = False
        self._current_index = 0
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"검사 진행률: 0 / {self._total_count}")
        self._set_status("대기")
        self._apply_idle_state()

    def _on_started(self, total: int) -> None:
        self._total_count = total
        self._current_index = 0
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"검사 진행률: 0 / {total}")
        self._set_status("검사 중")
        self._set_judgment(None, [], None)
        self.result_image_panel.clear_image("결과 이미지가 없습니다.")
        self.description.clear()
        self._apply_running_state()

    def _on_image_started(self, image_path: str, index: int, total: int) -> None:
        self._current_index = index
        self._total_count = total
        self.current_image_panel.set_image(Path(image_path))
        self.result_image_panel.clear_image("검사 중입니다. 결과 이미지는 완료 후 표시됩니다.")
        self._set_judgment(None, [], None)
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(max(0, index - 1))
        self.progress_label.setText(f"검사 진행률: {max(0, index - 1)} / {total}")
        self._set_status("검사 중")
        self.description.setPlainText(f"{Path(image_path).name} 검사 중...")

    def _on_result_ready(self, result: object) -> None:
        result_path = getattr(result, "result_image_path", None)
        if result_path:
            self.result_image_panel.set_image(Path(result_path))
        else:
            self.result_image_panel.clear_image("결과 이미지가 없습니다.")

        detections = list(getattr(result, "detections", []) or [])
        self._set_judgment(getattr(result, "status", None), detections, result)
        self.progress_bar.setValue(self._current_index)
        self.progress_label.setText(f"검사 진행률: {self._current_index} / {self._total_count}")
        self.description.setPlainText(_inspection_description_text(result, detections))

    def _on_finished(self) -> None:
        self._paused = False
        if self._stop_requested:
            self._stop_requested = False
            self._current_index = 0
            self.progress_bar.setValue(0)
            self.progress_label.setText(f"검사 진행률: 0 / {self._total_count}")
            self._set_status("대기")
        else:
            self.progress_bar.setValue(self._total_count)
            self.progress_label.setText(f"검사 진행률: {self._total_count} / {self._total_count}")
            self._set_status("완료")
        self._apply_idle_state()

    def _show_error(self, message: str) -> None:
        self._set_status("오류")
        QMessageBox.warning(self, "검사", message)

    def _apply_idle_state(self) -> None:
        self.start_button.setEnabled(True)
        self.start_button.setText("자동 검사 시작")
        self.pause_button.setEnabled(False)
        self.pause_button.setText("일시정지")
        self.stop_button.setEnabled(False)
        self.choose_button.setEnabled(True)

    def _apply_running_state(self) -> None:
        self.start_button.setEnabled(False)
        self.start_button.setText("자동 검사 시작")
        self.pause_button.setEnabled(True)
        self.pause_button.setText("일시정지")
        self.stop_button.setEnabled(True)
        self.choose_button.setEnabled(False)

    def _apply_paused_state(self) -> None:
        self.start_button.setEnabled(True)
        self.start_button.setText("검사 재개")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.choose_button.setEnabled(False)

    def _set_status(self, status: str) -> None:
        self.state_label.setText(f"현재 상태: {status}")

    def _set_judgment(self, status: object, detections: list[object], result: object | None) -> None:
        if status == "OK":
            self.judgment_label.setText("정상")
            self.judgment_label.setObjectName("JudgmentOk")
            defect_type = "-"
            confidence = "-"
        elif status == "NG":
            self.judgment_label.setText("불량")
            self.judgment_label.setObjectName("JudgmentNg")
            first_detection = detections[0] if detections else None
            defect_type = str(getattr(first_detection, "class_name", "-") or "-")
            confidence_value = getattr(first_detection, "confidence", None)
            confidence = f"{float(confidence_value) * 100:.1f}%" if confidence_value is not None else "-"
        else:
            self.judgment_label.setText("미판정")
            self.judgment_label.setObjectName("JudgmentNeutral")
            defect_type = "-"
            confidence = "-"

        self.defect_type_label.setText(defect_type)
        self.confidence_label.setText(confidence)
        self.judgment_label.style().unpolish(self.judgment_label)
        self.judgment_label.style().polish(self.judgment_label)


class ImagePanel(QWidget):
    def __init__(self, title: str, empty_text: str) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._empty_text = empty_text

        self.title_label = _section_title(title)
        self.image_label = FitImageLabel(empty_text)
        self.image_label.set_debug_name(title)
        self.image_label.setObjectName("ImageViewport")
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)

    def set_image(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.clear_image(f"이미지를 불러올 수 없습니다.\n{path}")
            return
        self._pixmap = pixmap
        self.image_label.set_original_pixmap(pixmap)

    def clear_image(self, text: str | None = None) -> None:
        self._pixmap = None
        self.image_label.clear_image(text or self._empty_text)


class FitImageLabel(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._original_pixmap: QPixmap | None = None
        self._debug_name = "image"
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_debug_name(self, name: str) -> None:
        self._debug_name = name

    def set_original_pixmap(self, pixmap: QPixmap) -> None:
        self._original_pixmap = pixmap
        self.setText("")
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        QTimer.singleShot(0, self._update_scaled_pixmap)

    def clear_image(self, text: str) -> None:
        self._original_pixmap = None
        self.clear()
        self.setText(text)
        self.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._original_pixmap is None:
            return
        target_size = self.contentsRect().size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            return
        scaled_pixmap = self._original_pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        print(
            "[IMAGE DEBUG]",
            self._debug_name,
            "original=", self._original_pixmap.size(),
            "label=", self.size(),
            "contents=", target_size,
            "scaled=", scaled_pixmap.size(),
        )
        self.setScaledContents(False)
        self.setPixmap(scaled_pixmap)
        self.setAlignment(Qt.AlignCenter)


def _section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    return label


def _value_panel(label: QLabel) -> QWidget:
    label.setObjectName("MetricValue")
    label.setAlignment(Qt.AlignCenter)
    panel = QWidget()
    panel.setObjectName("ValuePanel")
    panel.setMaximumHeight(50)
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(label)
    return panel


def _inspection_description_text(result: object, detections: list[object]) -> str:
    description = getattr(result, "vlm_description", None)
    if description:
        return str(description)
    if getattr(result, "status", None) == "OK":
        return "정상 이미지입니다. 탐지된 불량이 없습니다."

    lines = [
        "VLM 설명은 아직 생성되지 않았습니다.",
        "이력 화면에서 VLM 설명 생성을 실행할 수 있습니다.",
    ]
    inspection_id = getattr(result, "id", None)
    if inspection_id is not None:
        lines.append(f"저장된 검사 ID: {inspection_id}")
    if detections:
        lines.append("")
        lines.append("YOLO 탐지 정보:")
        for index, detection in enumerate(detections, start=1):
            class_name = getattr(detection, "class_name", "-")
            confidence = getattr(detection, "confidence", None)
            confidence_text = f"{float(confidence) * 100:.1f}%" if confidence is not None else "-"
            box = (
                getattr(detection, "x1", "-"),
                getattr(detection, "y1", "-"),
                getattr(detection, "x2", "-"),
                getattr(detection, "y2", "-"),
            )
            lines.append(
                f"{index}. {class_name} / {confidence_text} / bbox=({box[0]}, {box[1]}, {box[2]}, {box[3]})"
            )
    return "\n".join(lines)


def _inspection_stylesheet() -> str:
    return """
    QLabel#MutedText {
        color: #667085;
    }
    QLabel#SectionTitle {
        color: #17202a;
        font-size: 13px;
        font-weight: 700;
    }
    QLabel#ImageViewport {
        background: #ffffff;
        border: 1px solid #cfd7e3;
        border-radius: 6px;
        color: #7a8594;
        padding: 0;
    }
    QLabel#MetricValue {
        background: #ffffff;
        border: 1px solid #d6dde8;
        border-radius: 6px;
        color: #17202a;
        font-size: 14px;
        font-weight: 700;
        padding: 7px 10px;
    }
    QWidget#ValuePanel {
        background: transparent;
    }
    QLabel#JudgmentNeutral, QLabel#JudgmentOk, QLabel#JudgmentNg {
        border-radius: 6px;
        font-size: 22px;
        font-weight: 900;
        padding: 8px;
    }
    QLabel#JudgmentNeutral {
        background: #eef1f5;
        color: #667085;
        border: 1px solid #d6dde8;
    }
    QLabel#JudgmentOk {
        background: #e8f8ef;
        color: #147a42;
        border: 1px solid #8fd6ad;
    }
    QLabel#JudgmentNg {
        background: #fff0f1;
        color: #b4232d;
        border: 1px solid #f2a3aa;
    }
    QProgressBar {
        background: #ffffff;
        border: 1px solid #d6dde8;
        border-radius: 6px;
        color: #17202a;
        min-height: 14px;
        max-height: 16px;
        text-align: center;
    }
    QProgressBar::chunk {
        background: #2563eb;
        border-radius: 5px;
    }
    QPlainTextEdit {
        background: #ffffff;
        border: 1px solid #d6dde8;
        border-radius: 6px;
        color: #17202a;
        padding: 8px;
        selection-background-color: #bfdbfe;
    }
    """
