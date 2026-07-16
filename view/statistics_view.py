from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from viewmodel.statistics_viewmodel import StatisticsViewModel


class StatisticsView(QWidget):
    def __init__(self, viewmodel: StatisticsViewModel | None = None) -> None:
        super().__init__()
        self.viewmodel = viewmodel or StatisticsViewModel()
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("font-size: 20px; font-weight: 600;")
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.reload)
        top = QHBoxLayout()
        top.addWidget(self.summary_label, 1)
        top.addWidget(refresh_button)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Defect Type", "Count"])
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.table, 1)
        self.reload()

    def reload(self) -> None:
        summary = self.viewmodel.load_summary()
        ng_rate = (summary.ng_count / summary.total_count * 100) if summary.total_count else 0.0
        self.summary_label.setText(
            f"Total {summary.total_count} | OK {summary.ok_count} | "
            f"NG {summary.ng_count} | NG Rate {ng_rate:.1f}%"
        )
        items = list(summary.defect_type_counts.items())
        self.table.setRowCount(len(items))
        for row, (defect_type, count) in enumerate(items):
            self.table.setItem(row, 0, QTableWidgetItem(defect_type))
            self.table.setItem(row, 1, QTableWidgetItem(str(count)))
        self.table.resizeColumnsToContents()
