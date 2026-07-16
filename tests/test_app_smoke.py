import sys

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from view.main_window import MainWindow


def test_main_window_smoke(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)

    window = MainWindow()

    assert window.windowTitle() == "PCB Vision Inspection"
    assert window.centralWidget().count() == 4
    window.close()
