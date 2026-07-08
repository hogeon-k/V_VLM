from __future__ import annotations

from PySide6.QtWidgets import QWidget

from viewmodel.status_viewmodel import StatusViewModel


class StatusView(QWidget):
    def __init__(self, viewmodel: StatusViewModel | None = None) -> None:
        super().__init__()
        self.viewmodel = viewmodel or StatusViewModel()
        # TODO: Display model, database, and runtime status.
