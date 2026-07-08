from __future__ import annotations

from PySide6.QtWidgets import QWidget

from viewmodel.inspection_viewmodel import InspectionViewModel


class InspectionView(QWidget):
    def __init__(self, viewmodel: InspectionViewModel | None = None) -> None:
        super().__init__()
        self.viewmodel = viewmodel or InspectionViewModel()
        # TODO: Add image selection, preview, and inspection controls.
