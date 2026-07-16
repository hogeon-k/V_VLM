from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from view.history_view import HistoryView
from view.inspection_view import InspectionView
from view.statistics_view import StatisticsView
from view.status_view import StatusView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PCB Vision Inspection")
        self.setMinimumSize(1000, 700)

        tabs = QTabWidget()
        tabs.addTab(InspectionView(), "Inspection")
        tabs.addTab(HistoryView(), "History")
        tabs.addTab(StatisticsView(), "Statistics")
        tabs.addTab(StatusView(), "Status")
        self.setCentralWidget(tabs)
