from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox

from view.history_view import HistoryView
from view.image_viewer import ImageViewerDialog, ZoomableGraphicsView


class EmptyHistoryViewModel:
    def search(self, **_: object) -> list[object]:
        return []

    def defect_types(self) -> list[str]:
        return []

    def history_count(self) -> int:
        return 0


def _app(monkeypatch: pytest.MonkeyPatch) -> QApplication:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication(sys.argv)


def test_yolo_viewer_uses_original_pixmap(monkeypatch) -> None:
    _app(monkeypatch)
    opened: list[QPixmap] = []

    class FakeDialog:
        def __init__(self, *, pixmap: QPixmap, title: str, parent: object) -> None:
            opened.append(pixmap)
            assert title == "YOLO 결과 이미지 확대 보기"
            assert parent is not None

        def exec(self) -> int:
            return 0

    monkeypatch.setattr("view.history_view.ImageViewerDialog", FakeDialog)
    view = HistoryView(EmptyHistoryViewModel())  # type: ignore[arg-type]
    pixmap = QPixmap(640, 480)
    view.yolo_image._original_pixmap = pixmap

    view._open_yolo_image_viewer()

    assert opened == [pixmap]


def test_yolo_viewer_without_image_shows_message(monkeypatch) -> None:
    _app(monkeypatch)
    messages: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, _title, text: messages.append(text),
    )
    view = HistoryView(EmptyHistoryViewModel())  # type: ignore[arg-type]

    view._open_yolo_image_viewer()

    assert messages == ["확대할 YOLO 결과 이미지가 없습니다."]


def test_zoomable_view_respects_zoom_limits(monkeypatch) -> None:
    _app(monkeypatch)
    view = ZoomableGraphicsView(QPixmap(200, 100))

    view.reset_to_actual_size()
    for _ in range(30):
        view.zoom_by(1.25)
    assert view.current_scale <= view.maximum_scale

    for _ in range(60):
        view.zoom_by(0.8)
    assert view.current_scale >= view.minimum_scale


def test_image_viewer_dialog_has_controls(monkeypatch) -> None:
    _app(monkeypatch)
    dialog = ImageViewerDialog(QPixmap(200, 100), "테스트")

    assert dialog.windowTitle() == "테스트"
    assert dialog.minimumWidth() >= 900
    assert dialog.minimumHeight() >= 650
