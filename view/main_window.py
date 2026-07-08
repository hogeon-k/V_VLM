from __future__ import annotations

from PySide6.QtWidgets import QLabel, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PCB Vision Inspection")
        self.setMinimumSize(1000, 700)
        # TODO: Replace this placeholder with the MVVM-backed inspection UI.
        self.setCentralWidget(QLabel("PCB Vision Inspection"))
