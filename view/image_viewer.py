from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QPainter, QPixmap, QShortcut, QWheelEvent
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ZoomableGraphicsView(QGraphicsView):
    zoom_changed = Signal(float)

    def __init__(self, pixmap: QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.minimum_scale = 0.10
        self.maximum_scale = 8.0
        self.current_scale = 1.0
        self._fit_mode = True
        self._scene = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self.pixmap_item)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )

    def fit_image(self) -> None:
        self._fit_mode = True
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.current_scale = self.transform().m11()
        self.zoom_changed.emit(self.current_scale)

    def reset_to_actual_size(self) -> None:
        self._fit_mode = False
        self.resetTransform()
        self.current_scale = 1.0
        self.zoom_changed.emit(self.current_scale)

    def zoom_by(self, factor: float) -> None:
        new_scale = self.current_scale * factor
        if new_scale < self.minimum_scale or new_scale > self.maximum_scale:
            return
        self._fit_mode = False
        self.scale(factor, factor)
        self.current_scale = new_scale
        self.zoom_changed.emit(self.current_scale)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.zoom_by(factor)

    def mouseDoubleClickEvent(self, event: object) -> None:
        if self._fit_mode:
            self.reset_to_actual_size()
        else:
            self.fit_image()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_image()


class ImageViewerDialog(QDialog):
    def __init__(self, pixmap: QPixmap, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 650)
        self.view = ZoomableGraphicsView(pixmap, self)
        self.zoom_label = QLabel("100%")

        fit_button = QPushButton("전체 맞춤")
        actual_button = QPushButton("100%")
        zoom_out_button = QPushButton("축소")
        zoom_in_button = QPushButton("확대")
        close_button = QPushButton("닫기")

        fit_button.clicked.connect(self.view.fit_image)
        actual_button.clicked.connect(self.view.reset_to_actual_size)
        zoom_out_button.clicked.connect(lambda: self.view.zoom_by(0.8))
        zoom_in_button.clicked.connect(lambda: self.view.zoom_by(1.25))
        close_button.clicked.connect(self.accept)
        self.view.zoom_changed.connect(self._update_zoom_label)

        controls = QHBoxLayout()
        controls.addWidget(fit_button)
        controls.addWidget(actual_button)
        controls.addWidget(zoom_out_button)
        controls.addWidget(zoom_in_button)
        controls.addStretch(1)
        controls.addWidget(self.zoom_label)
        controls.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.view, 1)

        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.accept)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self.view.fit_image)
        QShortcut(QKeySequence("Ctrl+1"), self, activated=self.view.reset_to_actual_size)
        QShortcut(QKeySequence("+"), self, activated=lambda: self.view.zoom_by(1.25))
        QShortcut(QKeySequence("Ctrl++"), self, activated=lambda: self.view.zoom_by(1.25))
        QShortcut(QKeySequence("-"), self, activated=lambda: self.view.zoom_by(0.8))
        QShortcut(QKeySequence("Ctrl+-"), self, activated=lambda: self.view.zoom_by(0.8))
        self._resize_to_parent(parent)

    def showEvent(self, event: object) -> None:
        super().showEvent(event)
        self.view.fit_image()

    def _update_zoom_label(self, scale: float) -> None:
        self.zoom_label.setText(f"{scale * 100:.0f}%")

    def _resize_to_parent(self, parent: QWidget | None) -> None:
        if parent is None or parent.window() is None:
            return
        size = parent.window().size()
        width = max(900, int(size.width() * 0.8))
        height = max(650, int(size.height() * 0.8))
        self.resize(width, height)
