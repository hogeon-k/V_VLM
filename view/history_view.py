from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QDate, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from view.image_viewer import ImageViewerDialog
from viewmodel.history_viewmodel import HistoryViewModel


class HistoryView(QWidget):
    def __init__(self, viewmodel: HistoryViewModel | None = None) -> None:
        super().__init__()
        self.viewmodel = viewmodel or HistoryViewModel()
        self.results: list[object] = []
        self._has_any_records = False

        self.all_dates_checkbox = QCheckBox("전체 기간")
        self.all_dates_checkbox.setChecked(True)
        self.all_dates_checkbox.toggled.connect(self._sync_date_filter_state)

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addMonths(-1))
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())

        self.status_filter = QComboBox()
        self.status_filter.addItems(["전체", "정상", "불량", "미판정"])
        self.defect_filter = QComboBox()
        self.defect_filter.addItem("전체")

        self.refresh_button = QPushButton("검색")
        self.refresh_button.clicked.connect(self.reload)
        self.reset_button = QPushButton("초기화")
        self.reset_button.clicked.connect(self.reset_filters)

        self.delete_selected_button = QPushButton("선택 기록 삭제")
        self.delete_selected_button.setObjectName("DangerButton")
        self.delete_selected_button.setEnabled(False)
        self.delete_selected_button.clicked.connect(self._delete_selected)
        self.delete_all_button = QPushButton("전체 기록 삭제")
        self.delete_all_button.setObjectName("DangerButton")
        self.delete_all_button.setEnabled(False)
        self.delete_all_button.clicked.connect(self._delete_all)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["번호", "검사 일시", "이미지명", "판정", "불량 유형", "신뢰도"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._show_selected_detail)
        self.table.itemSelectionChanged.connect(self._update_delete_buttons)

        self.count_label = QLabel("총 0건")
        self.count_label.setObjectName("MutedText")
        self.state_label = QLabel("")
        self.state_label.setObjectName("MutedText")

        self.original_image = FitImageLabel("검사 이력을 선택하면 원본 이미지가 표시됩니다.")
        self.yolo_image = FitImageLabel("검사 이력을 선택하면 YOLO 결과 이미지가 표시됩니다.")
        self.yolo_image.setCursor(Qt.CursorShape.PointingHandCursor)
        self.yolo_image.setToolTip("클릭하면 확대해서 볼 수 있습니다.")
        self.yolo_image.clicked.connect(self._open_yolo_image_viewer)
        self.status_value = QLabel("-")
        self.defect_value = QLabel("-")
        self.confidence_value = QLabel("-")
        self.inspected_at_value = QLabel("-")
        self.image_name_value = QLabel("-")
        self.vlm_text = QPlainTextEdit()
        self.vlm_text.setReadOnly(True)
        self.vlm_text.setPlainText("검사 이력을 선택하면 상세 결과가 표시됩니다.")
        self.detail = self.vlm_text

        self._build_layout()
        self.setStyleSheet(_history_stylesheet())
        self._sync_date_filter_state()
        self.reload()

    def _build_layout(self) -> None:
        filters = QHBoxLayout()
        filters.setSpacing(8)
        filters.addWidget(self.all_dates_checkbox)
        filters.addWidget(QLabel("시작일"))
        filters.addWidget(self.start_date)
        filters.addWidget(QLabel("종료일"))
        filters.addWidget(self.end_date)
        filters.addWidget(QLabel("결과"))
        filters.addWidget(self.status_filter)
        filters.addWidget(QLabel("불량 유형"))
        filters.addWidget(self.defect_filter)
        filters.addWidget(self.refresh_button)
        filters.addWidget(self.reset_button)
        filters.addStretch(1)

        history_panel = QWidget()
        history_layout = QVBoxLayout(history_panel)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(8)
        history_layout.addWidget(_section_title("검사 이력 목록"))
        history_layout.addWidget(self.table, 1)
        history_layout.addWidget(self.count_label)

        detail_panel = self._build_detail_panel()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(history_panel)
        splitter.addWidget(detail_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([720, 480])

        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(self.delete_selected_button)
        actions.addWidget(self.delete_all_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 8, 24, 16)
        layout.setSpacing(10)
        layout.addLayout(filters)
        layout.addWidget(self.state_label)
        layout.addWidget(splitter, 1)
        layout.addLayout(actions)

    def _build_detail_panel(self) -> QScrollArea:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(_section_title("상세 결과"))

        image_grid = QGridLayout()
        image_grid.setHorizontalSpacing(10)
        image_grid.setVerticalSpacing(6)
        image_grid.addWidget(_section_title("원본 이미지"), 0, 0)
        image_grid.addWidget(_section_title("YOLO 결과 이미지"), 0, 1)
        image_grid.addWidget(self.original_image, 1, 0)
        image_grid.addWidget(self.yolo_image, 1, 1)
        image_grid.setColumnStretch(0, 1)
        image_grid.setColumnStretch(1, 1)
        layout.addLayout(image_grid, 2)

        info_group = QGroupBox("판정 정보")
        form = QFormLayout(info_group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow("판정 결과", self.status_value)
        form.addRow("불량 유형", self.defect_value)
        form.addRow("신뢰도", self.confidence_value)
        form.addRow("검사 시간", self.inspected_at_value)
        form.addRow("이미지명", self.image_name_value)
        layout.addWidget(info_group)

        layout.addWidget(_section_title("VLM 설명"))
        layout.addWidget(self.vlm_text, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        return scroll

    def reload(self) -> None:
        if self.start_date.date() > self.end_date.date() and not self.all_dates_checkbox.isChecked():
            QMessageBox.warning(self, "검사 이력 검색", "시작일은 종료일보다 늦을 수 없습니다.")
            return

        self.state_label.setText("조회 중...")
        self._reload_defect_types()
        defect_type = self.defect_filter.currentText()
        self.results = self.viewmodel.search(
            start_date=None if self.all_dates_checkbox.isChecked() else self.start_date.date().toString("yyyy-MM-dd"),
            end_date=None if self.all_dates_checkbox.isChecked() else self.end_date.date().toString("yyyy-MM-dd"),
            status=_status_to_db(self.status_filter.currentText()),
            defect_type=None if defect_type in ("", "전체") else defect_type,
        )
        self._populate_table()
        self._has_any_records = self.viewmodel.history_count() > 0
        self._clear_detail(clear_selection=True)
        self._update_delete_buttons()
        if self.results:
            self.state_label.setText(f"총 {len(self.results)}건")
        else:
            self.state_label.setText("검색 결과가 없습니다.")

    def reset_filters(self) -> None:
        self.all_dates_checkbox.setChecked(True)
        self.start_date.setDate(QDate.currentDate().addMonths(-1))
        self.end_date.setDate(QDate.currentDate())
        self.status_filter.setCurrentIndex(0)
        self.defect_filter.setCurrentIndex(0)
        self.reload()

    def _populate_table(self) -> None:
        self.table.setSortingEnabled(False)
        self.table.clearSelection()
        self.table.setRowCount(len(self.results))
        display_numbers = _display_numbers_by_inspection_id(self.results)
        for row_index, result in enumerate(self.results):
            inspection_id = getattr(result, "id", None)
            first_defect = _first_defect(result)
            values = [
                display_numbers.get(int(inspection_id), row_index + 1) if inspection_id is not None else row_index + 1,
                _dt_text(getattr(result, "inspected_at", None)),
                getattr(result, "image_name", ""),
                _status_text(getattr(result, "status", "")),
                _defect_text(first_defect),
                _confidence_text(first_defect),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem()
                item.setData(Qt.ItemDataRole.DisplayRole, int(value) if col == 0 else str(value))
                if inspection_id is not None:
                    item.setData(Qt.ItemDataRole.UserRole, int(inspection_id))
                self.table.setItem(row_index, col, item)
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)
        self.table.sortItems(0, Qt.SortOrder.AscendingOrder)
        self.count_label.setText(f"총 {len(self.results)}건")

    def _reload_defect_types(self) -> None:
        current = self.defect_filter.currentText() or "전체"
        self.defect_filter.blockSignals(True)
        self.defect_filter.clear()
        self.defect_filter.addItem("전체")
        self.defect_filter.addItems(self.viewmodel.defect_types())
        index = self.defect_filter.findText(current)
        self.defect_filter.setCurrentIndex(index if index >= 0 else 0)
        self.defect_filter.blockSignals(False)

    def _show_selected_detail(self) -> None:
        inspection_id = self._selected_inspection_id()
        if inspection_id is None:
            self._clear_detail(clear_selection=False)
            return
        self._clear_detail(clear_selection=False)
        try:
            result = self.viewmodel.load_detail(inspection_id)
        except Exception as exc:
            QMessageBox.critical(self, "검사 이력 상세", f"DB 조회 실패: {exc}")
            return
        if result is None:
            QMessageBox.warning(self, "검사 이력 상세", "검사 기록을 찾을 수 없습니다.")
            return
        self._render_detail(result)

    def _render_detail(self, result: object) -> None:
        self.original_image.set_image_path(
            getattr(result, "original_image_path", None),
            missing_text="원본 이미지를 찾을 수 없습니다.",
        )
        self.yolo_image.set_image_path(
            getattr(result, "result_image_path", None),
            missing_text="YOLO 결과 이미지가 없습니다.",
        )

        status = str(getattr(result, "status", "") or "")
        first_defect = _first_defect(result)
        self.status_value.setText(_status_text(status))
        self.status_value.setProperty("status", status)
        self.status_value.style().unpolish(self.status_value)
        self.status_value.style().polish(self.status_value)
        self.defect_value.setText(_defect_text(first_defect))
        self.confidence_value.setText(_confidence_text(first_defect))
        self.inspected_at_value.setText(_dt_text(getattr(result, "inspected_at", None)) or "-")
        self.image_name_value.setText(str(getattr(result, "image_name", "") or "-"))
        self.vlm_text.setPlainText(getattr(result, "vlm_description", None) or "VLM 분석 결과가 없습니다.")

    def _selected_inspection_id(self) -> int | None:
        if not self.table.selectedItems():
            return None
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return int(value) if value is not None else None

    def _delete_selected(self) -> None:
        inspection_id = self._selected_inspection_id()
        if inspection_id is None:
            QMessageBox.information(self, "검사 기록 삭제", "삭제할 검사 기록을 선택해주세요.")
            return

        answer = QMessageBox.question(
            self,
            "검사 기록 삭제",
            "선택한 검사 기록을 삭제하시겠습니까?\n\n"
            "연결된 검사 결과와 프로그램이 생성한 결과 이미지도 함께 삭제됩니다.\n"
            "이 작업은 되돌릴 수 없습니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        try:
            report = self.viewmodel.delete_history(inspection_id)
        except ValueError as exc:
            QMessageBox.warning(self, "검사 기록 삭제", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "검사 기록 삭제", f"DB 삭제 실패: {exc}")
            return

        self.reload()
        self._show_deletion_success("선택한 검사 기록이 삭제되었습니다.", report)

    def _delete_all(self) -> None:
        answer = QMessageBox.question(
            self,
            "전체 검사 기록 삭제",
            "저장된 모든 검사 기록을 삭제하시겠습니까?\n\n"
            "모든 검사 결과와 프로그램이 생성한 결과 이미지가 삭제됩니다.\n"
            "이 작업은 되돌릴 수 없습니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        text, accepted = QInputDialog.getText(
            self,
            "전체 검사 기록 삭제",
            "계속하려면 '전체 삭제'를 입력하세요.",
        )
        if not accepted or text != "전체 삭제":
            return

        try:
            report = self.viewmodel.delete_all_history()
        except Exception as exc:
            QMessageBox.critical(self, "전체 검사 기록 삭제", f"DB 삭제 실패: {exc}")
            return

        self.reload()
        self._show_deletion_success("전체 검사 기록이 삭제되었습니다.", report)

    def _show_deletion_success(self, message: str, report: object) -> None:
        if getattr(report, "has_file_warnings", False):
            QMessageBox.warning(
                self,
                "검사 기록 삭제",
                f"{message}\n\n일부 이미지 파일 정리가 실패했거나 허용 폴더 밖이라 삭제하지 않았습니다.",
            )
            return
        QMessageBox.information(self, "검사 기록 삭제", message)

    def _open_yolo_image_viewer(self) -> None:
        pixmap = self.yolo_image.original_pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.information(
                self,
                "이미지 없음",
                "확대할 YOLO 결과 이미지가 없습니다.",
            )
            return
        dialog = ImageViewerDialog(
            pixmap=pixmap,
            title="YOLO 결과 이미지 확대 보기",
            parent=self,
        )
        dialog.exec()

    def _clear_detail(self, *, clear_selection: bool) -> None:
        if clear_selection:
            self.table.clearSelection()
        self.original_image.clear_image("검사 이력을 선택하면 원본 이미지가 표시됩니다.")
        self.yolo_image.clear_image("검사 이력을 선택하면 YOLO 결과 이미지가 표시됩니다.")
        self.status_value.setText("-")
        self.status_value.setProperty("status", "")
        self.status_value.style().unpolish(self.status_value)
        self.status_value.style().polish(self.status_value)
        self.defect_value.setText("-")
        self.confidence_value.setText("-")
        self.inspected_at_value.setText("-")
        self.image_name_value.setText("-")
        self.vlm_text.setPlainText("검사 이력을 선택하면 상세 결과가 표시됩니다.")

    def _update_delete_buttons(self) -> None:
        self.delete_selected_button.setEnabled(self._selected_inspection_id() is not None)
        self.delete_all_button.setEnabled(self._has_any_records)

    def _sync_date_filter_state(self) -> None:
        enabled = not self.all_dates_checkbox.isChecked()
        self.start_date.setEnabled(enabled)
        self.end_date.setEnabled(enabled)


class FitImageLabel(QLabel):
    clicked = Signal()

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._original_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("ImagePreview")

    def original_pixmap(self) -> QPixmap | None:
        return self._original_pixmap

    def set_image_path(self, path: object, *, missing_text: str) -> None:
        if not path:
            self.clear_image(missing_text)
            return
        image_path = Path(path)
        if not image_path.is_file():
            self.clear_image(missing_text)
            return
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.clear_image(missing_text)
            return
        self._original_pixmap = pixmap
        self.setText("")
        QTimer.singleShot(0, self._refresh_pixmap)

    def clear_image(self, text: str) -> None:
        self._original_pixmap = None
        self.clear()
        self.setText(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def mousePressEvent(self, event: object) -> None:
        if hasattr(event, "button") and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def sizeHint(self) -> QSize:
        return QSize(260, 180)

    def _refresh_pixmap(self) -> None:
        if self._original_pixmap is None:
            return
        target_size = self.contentsRect().size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            return
        scaled = self._original_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setScaledContents(False)
        self.setPixmap(scaled)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


def _section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    return label


def _first_defect(result: object) -> object | None:
    defects = list(getattr(result, "defects", getattr(result, "detections", [])) or [])
    return defects[0] if defects else None


def _display_numbers_by_inspection_id(results: list[object]) -> dict[int, int]:
    numbered_results = [
        result
        for result in results
        if getattr(result, "id", None) is not None
    ]
    sorted_results = sorted(
        numbered_results,
        key=lambda result: (
            _dt_text(getattr(result, "inspected_at", None)),
            int(getattr(result, "id")),
        ),
    )
    return {
        int(getattr(result, "id")): index + 1
        for index, result in enumerate(sorted_results)
    }


def _defect_text(defect: object | None) -> str:
    if defect is None:
        return "-"
    return str(getattr(defect, "defect_type", getattr(defect, "class_name", "-")) or "-")


def _confidence_text(defect: object | None) -> str:
    if defect is None:
        return "-"
    value = getattr(defect, "confidence", None)
    if value is None:
        return "-"
    numeric = float(value)
    percent = numeric * 100 if numeric <= 1 else numeric
    return f"{percent:.1f}%"


def _status_text(status: object) -> str:
    if status == "OK":
        return "정상"
    if status == "NG":
        return "불량"
    if status == "PENDING":
        return "미판정"
    return str(status or "미판정")


def _status_to_db(text: str) -> str | None:
    return {
        "전체": "ALL",
        "정상": "OK",
        "불량": "NG",
        "미판정": "PENDING",
    }.get(text, "ALL")


def _dt_text(value: object) -> str:
    return value.isoformat(sep=" ", timespec="seconds") if hasattr(value, "isoformat") else ""


def _history_stylesheet() -> str:
    return """
    QLabel#MutedText {
        color: #667085;
    }
    QLabel#SectionTitle {
        color: #17202a;
        font-size: 13px;
        font-weight: 700;
    }
    QLabel#ImagePreview {
        background: #ffffff;
        border: 1px solid #cfd7e3;
        border-radius: 6px;
        color: #7a8594;
        padding: 0;
    }
    QGroupBox {
        background: #ffffff;
        border: 1px solid #d6dde8;
        border-radius: 6px;
        margin-top: 12px;
        padding: 10px;
        font-weight: 700;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
    }
    QPlainTextEdit, QTableWidget {
        background: #ffffff;
        border: 1px solid #d6dde8;
        border-radius: 6px;
        color: #17202a;
        selection-background-color: #bfdbfe;
    }
    QTableWidget::item {
        padding: 5px;
    }
    QLabel[status="OK"] {
        color: #147a42;
        font-weight: 800;
    }
    QLabel[status="NG"] {
        color: #b4232d;
        font-weight: 800;
    }
    QLabel[status="PENDING"] {
        color: #667085;
        font-weight: 800;
    }
    QPushButton#DangerButton {
        background: #fff1f2;
        color: #b4232d;
        border: 1px solid #f1a0a8;
        border-radius: 6px;
        padding: 7px 12px;
        font-weight: 700;
    }
    QPushButton#DangerButton:hover {
        background: #ffe4e6;
        border-color: #e11d48;
    }
    QPushButton#DangerButton:disabled {
        background: #f2f4f7;
        color: #98a2b3;
        border-color: #d0d5dd;
    }
    """
