from __future__ import annotations

from PySide6.QtWidgets import QWidget

from viewmodel.statistics_viewmodel import StatisticsViewModel


class StatisticsView(QWidget):
    def __init__(self, viewmodel: StatisticsViewModel | None = None) -> None:
        super().__init__()
        self.viewmodel = viewmodel or StatisticsViewModel()
        # TODO: Display aggregated inspection statistics.
