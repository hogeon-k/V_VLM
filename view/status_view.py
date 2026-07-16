from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from viewmodel.status_viewmodel import StatusViewModel


class StatusView(QWidget):
    def __init__(self, viewmodel: StatusViewModel | None = None) -> None:
        super().__init__()
        self.viewmodel = viewmodel or StatusViewModel()
        self.form = QFormLayout()
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.reload)
        layout = QVBoxLayout(self)
        layout.addLayout(self.form)
        layout.addWidget(refresh_button)
        layout.addStretch(1)
        self.reload()

    def reload(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        for key, value in self.viewmodel.get_status().items():
            self.form.addRow(QLabel(key), QLabel(value))
