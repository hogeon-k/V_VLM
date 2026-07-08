from __future__ import annotations

from PySide6.QtWidgets import QWidget

from viewmodel.history_viewmodel import HistoryViewModel


class HistoryView(QWidget):
    def __init__(self, viewmodel: HistoryViewModel | None = None) -> None:
        super().__init__()
        self.viewmodel = viewmodel or HistoryViewModel()
        # TODO: Display persisted inspection history.
