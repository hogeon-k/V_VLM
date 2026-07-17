from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from viewmodel.status_viewmodel import StatusViewModel


class StatusView(QWidget):
    def __init__(self, viewmodel: StatusViewModel | None = None) -> None:
        super().__init__()
        self.viewmodel = viewmodel or StatusViewModel()
        self._vlm_thread: QThread | None = None
        self._vlm_worker: VlmStatusWorker | None = None

        self.refresh_button = QPushButton("상태 새로고침")
        self.refresh_button.setObjectName("RefreshButton")
        self.refresh_button.clicked.connect(self.reload)

        self.yolo_card = StatusCard("YOLO 모델 상태")
        self.vlm_card = StatusCard("VLM 상태")
        self.database_card = StatusCard("SQL DB 상태")
        self.gpu_card = StatusCard("GPU 상태")
        self.log_card = LogCard("로그 상태")

        self._build_layout()
        self.setStyleSheet(_status_stylesheet())
        self.reload()

    def _build_layout(self) -> None:
        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 8, 12, 8)
        top_layout.addWidget(self.refresh_button)
        top_layout.addStretch(1)

        cards = QGridLayout()
        cards.setHorizontalSpacing(28)
        cards.setVerticalSpacing(28)
        cards.addWidget(self.yolo_card, 0, 0)
        cards.addWidget(self.vlm_card, 0, 1)
        cards.addWidget(self.database_card, 1, 0)
        cards.addWidget(self.gpu_card, 1, 1)
        cards.setColumnStretch(0, 1)
        cards.setColumnStretch(1, 1)
        cards.setRowStretch(0, 1)
        cards.setRowStretch(1, 1)

        cards_panel = QFrame()
        cards_panel.setObjectName("Panel")
        cards_panel_layout = QVBoxLayout(cards_panel)
        cards_panel_layout.setContentsMargins(24, 24, 24, 24)
        cards_panel_layout.addLayout(cards)

        bottom_panel = QFrame()
        bottom_panel.setObjectName("Panel")
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(24, 18, 24, 18)
        bottom_layout.addWidget(self.log_card, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 8, 24, 16)
        layout.setSpacing(16)
        layout.addWidget(top_bar)
        layout.addWidget(cards_panel, 4)
        layout.addWidget(bottom_panel, 1)

    def reload(self) -> None:
        status = self.viewmodel.get_status()
        self.yolo_card.set_status(
            status.get("yolo_model", "Unknown"),
            status.get("model_path", ""),
        )
        self.vlm_card.set_status(
            status.get("vlm", "Unknown"),
            status.get("vlm_detail", ""),
        )
        self.database_card.set_status(
            status.get("database", "Unknown"),
            status.get("database_path", ""),
        )
        self.gpu_card.set_status(
            status.get("gpu", "Unknown"),
            "CUDA / NVIDIA runtime visibility",
        )
        self.log_card.set_status(
            status.get("logs", "Unknown"),
            status.get("log_path", ""),
            status.get("recent_logs", "아직 기록된 실행 로그가 없습니다."),
        )
        self._start_vlm_status_check()

    def _start_vlm_status_check(self) -> None:
        self.vlm_card.set_status("확인 중...", "Ollama 서버 상태를 확인하는 중입니다.")
        if self._vlm_thread is not None and self._vlm_thread.isRunning():
            return
        self._vlm_thread = QThread(self)
        self._vlm_worker = VlmStatusWorker(self.viewmodel)
        self._vlm_worker.moveToThread(self._vlm_thread)
        self._vlm_thread.started.connect(self._vlm_worker.run)
        self._vlm_worker.finished.connect(self._on_vlm_status_finished)
        self._vlm_worker.finished.connect(self._vlm_thread.quit)
        self._vlm_worker.finished.connect(self._vlm_worker.deleteLater)
        self._vlm_thread.finished.connect(self._vlm_thread.deleteLater)
        self._vlm_thread.finished.connect(self._clear_vlm_worker_refs)
        self._vlm_thread.start()

    def closeEvent(self, event: object) -> None:
        self.stop_vlm_status_check()
        super().closeEvent(event)

    def stop_vlm_status_check(self) -> None:
        if self._vlm_thread is not None and self._vlm_thread.isRunning():
            self._vlm_thread.quit()
            self._vlm_thread.wait(3000)

    @Slot(str, str)
    def _on_vlm_status_finished(self, state: str, detail: str) -> None:
        self.vlm_card.set_status(state, detail)
        self.vlm_card.setToolTip(detail)

    @Slot()
    def _clear_vlm_worker_refs(self) -> None:
        self._vlm_thread = None
        self._vlm_worker = None


class VlmStatusWorker(QObject):
    finished = Signal(str, str)

    def __init__(self, viewmodel: StatusViewModel) -> None:
        super().__init__()
        self.viewmodel = viewmodel

    @Slot()
    def run(self) -> None:
        try:
            status = self.viewmodel.check_vlm_status()
            detail = f"Host: {status.host}\nModel: {status.model_name}\n{status.detail}"
            self.finished.emit(status.state, detail)
        except Exception as exc:
            self.finished.emit("연결 실패", str(exc))


class StatusCard(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("StatusCard")
        self.title_label = QLabel(title)
        self.title_label.setObjectName("StatusTitle")
        self.state_label = QLabel("-")
        self.state_label.setObjectName("StatusValue")
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("StatusDetail")
        self.detail_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        layout.addStretch(1)
        layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.state_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.detail_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)

    def set_status(self, state: str, detail: str) -> None:
        self.state_label.setText(state)
        self.detail_label.setText(detail)
        self.setProperty("state", _state_property(state))
        self.style().unpolish(self)
        self.style().polish(self)


class LogCard(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("StatusCard")
        self.title_label = QLabel(title)
        self.title_label.setObjectName("StatusTitle")
        self.state_label = QLabel("-")
        self.state_label.setObjectName("StatusValue")
        self.path_label = QLabel("")
        self.path_label.setObjectName("StatusDetail")
        self.path_label.setWordWrap(True)
        self.log_text = QPlainTextEdit()
        self.log_text.setObjectName("LogText")
        self.log_text.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)
        header = QHBoxLayout()
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.state_label)
        layout.addLayout(header)
        layout.addWidget(self.path_label)
        layout.addWidget(self.log_text, 1)

    def set_status(self, state: str, path: str, recent_logs: str) -> None:
        self.state_label.setText(state)
        self.path_label.setText(path)
        self.log_text.setPlainText(recent_logs)
        self.setProperty("state", _state_property(state))
        self.style().unpolish(self)
        self.style().polish(self)


def _state_property(value: str) -> str:
    lowered = value.lower()
    if any(term in lowered for term in ("ready", "configured", "visible", "연결됨")):
        return "ok"
    if any(term in lowered for term in ("missing", "error", "fail", "없음", "실패", "오류")):
        return "error"
    return "neutral"


def _status_stylesheet() -> str:
    return """
    QFrame#TopBar, QFrame#Panel {
        background: #ffffff;
        border: 1px solid #d6dde8;
        border-radius: 6px;
    }
    QPushButton#RefreshButton {
        background: #ffffff;
        color: #263241;
        border: 1px solid #ccd4df;
        border-radius: 6px;
        padding: 8px 14px;
        font-weight: 700;
    }
    QPushButton#RefreshButton:hover {
        background: #eef4ff;
        border-color: #8fb5ff;
    }
    QFrame#StatusCard {
        background: #ffffff;
        border: 1px solid #cfd7e3;
        border-radius: 6px;
    }
    QFrame#StatusCard[state="ok"] {
        border-color: #8fd6ad;
    }
    QFrame#StatusCard[state="error"] {
        border-color: #f2a3aa;
    }
    QLabel#StatusTitle {
        color: #17202a;
        font-size: 14px;
        font-weight: 800;
    }
    QLabel#StatusValue {
        color: #263241;
        font-size: 22px;
        font-weight: 900;
    }
    QFrame#StatusCard[state="ok"] QLabel#StatusValue {
        color: #147a42;
    }
    QFrame#StatusCard[state="error"] QLabel#StatusValue {
        color: #b4232d;
    }
    QLabel#StatusDetail {
        color: #667085;
        font-size: 12px;
    }
    QPlainTextEdit#LogText {
        background: #f8fafc;
        border: 1px solid #d6dde8;
        border-radius: 6px;
        color: #17202a;
        font-family: "Consolas", "Segoe UI", monospace;
        font-size: 12px;
        padding: 8px;
    }
    """
