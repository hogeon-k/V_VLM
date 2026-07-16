from __future__ import annotations

from PySide6.QtCore import QDate
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from viewmodel.history_viewmodel import HistoryViewModel


class HistoryView(QWidget):
    def __init__(self, viewmodel: HistoryViewModel | None = None) -> None:
        super().__init__()
        self.viewmodel = viewmodel or HistoryViewModel()
        self.results: list[object] = []

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addMonths(-1))
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate().addDays(1))
        self.status_filter = QComboBox()
        self.status_filter.addItems(["ALL", "OK", "NG"])
        self.defect_filter = QComboBox()
        self.defect_filter.addItem("")
        refresh_button = QPushButton("Search")
        refresh_button.clicked.connect(self.reload)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("From"))
        filters.addWidget(self.start_date)
        filters.addWidget(QLabel("To"))
        filters.addWidget(self.end_date)
        filters.addWidget(QLabel("Status"))
        filters.addWidget(self.status_filter)
        filters.addWidget(QLabel("Defect"))
        filters.addWidget(self.defect_filter)
        filters.addWidget(refresh_button)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "ID", "Time", "Image", "Status", "Defects", "Result", "VLM"
        ])
        self.table.itemSelectionChanged.connect(self._show_selected_detail)
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addLayout(filters)
        layout.addWidget(self.table, 2)
        layout.addWidget(QLabel("Detail"))
        layout.addWidget(self.detail, 1)
        self.reload()

    def reload(self) -> None:
        self._reload_defect_types()
        defect_type = self.defect_filter.currentText() or None
        self.results = self.viewmodel.search(
            start_date=self.start_date.date().toString("yyyy-MM-dd"),
            end_date=self.end_date.date().toString("yyyy-MM-dd"),
            status=self.status_filter.currentText(),
            defect_type=defect_type,
        )
        self.table.setRowCount(len(self.results))
        for row_index, result in enumerate(self.results):
            values = [
                getattr(result, "id", "") or "",
                _dt_text(getattr(result, "inspected_at", None)),
                getattr(result, "image_name", ""),
                getattr(result, "status", ""),
                getattr(result, "defect_count", ""),
                str(getattr(result, "result_image_path", "") or ""),
                (getattr(result, "vlm_description", "") or "")[:80],
            ]
            for col, value in enumerate(values):
                self.table.setItem(row_index, col, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()

    def _reload_defect_types(self) -> None:
        current = self.defect_filter.currentText()
        self.defect_filter.blockSignals(True)
        self.defect_filter.clear()
        self.defect_filter.addItem("")
        self.defect_filter.addItems(self.viewmodel.defect_types())
        index = self.defect_filter.findText(current)
        if index >= 0:
            self.defect_filter.setCurrentIndex(index)
        self.defect_filter.blockSignals(False)

    def _show_selected_detail(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.results):
            return
        result = self.results[row]
        defect_lines = [
            f"- {d.defect_type} {d.confidence:.3f} ({d.bbox_x1}, {d.bbox_y1}, {d.bbox_x2}, {d.bbox_y2})"
            for d in getattr(result, "defects", [])
        ]
        self.detail.setPlainText(
            "\n".join([
                f"ID: {getattr(result, 'id', '')}",
                f"Image: {getattr(result, 'original_image_path', '')}",
                f"Result: {getattr(result, 'result_image_path', '')}",
                f"Status: {getattr(result, 'status', '')}",
                f"Inspected At: {_dt_text(getattr(result, 'inspected_at', None))}",
                "Defects:",
                *(defect_lines or ["- none"]),
                "",
                "VLM:",
                getattr(result, "vlm_description", None) or "",
            ])
        )


def _dt_text(value: object) -> str:
    return value.isoformat(sep=" ", timespec="seconds") if hasattr(value, "isoformat") else ""
