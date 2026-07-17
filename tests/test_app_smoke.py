import sys

import pytest

pytest.importorskip("PySide6")
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from view.inspection_view import FitImageLabel
from view.main_window import MainWindow


def test_main_window_smoke(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)

    window = MainWindow()

    assert window.windowTitle() == "PCB Vision Inspection"
    assert window.stack.count() == 4
    assert window.stack.currentIndex() == 0
    window.close()


def test_fit_image_label_scales_pixmap_inside_contents(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)

    label = FitImageLabel("empty")
    label.resize(320, 180)
    original = QPixmap(800, 600)

    label.set_original_pixmap(original)
    app.processEvents()

    rendered = label.pixmap()
    contents_size = label.contentsRect().size()

    assert rendered is not None
    assert rendered.width() <= contents_size.width()
    assert rendered.height() <= contents_size.height()
    assert rendered.size() != contents_size
