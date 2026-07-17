from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from config import settings
from service.ollama_status_service import OllamaStatus
from view.status_view import StatusView
from viewmodel.status_viewmodel import StatusViewModel


class FakeStatusViewModel:
    def __init__(self) -> None:
        self.calls = 0
        self.vlm_calls = 0

    def get_status(self) -> dict[str, str]:
        self.calls += 1
        return {
            "model_path": "models/best.pt",
            "yolo_model": "Ready",
            "vlm": "Configured for local Ollama",
            "database_path": "database/inspection_results.sqlite3",
            "database": "Ready",
            "gpu": "NVIDIA driver visible",
            "log_path": "logs/error.log",
            "logs": "Ready",
            "recent_logs": "2026-07-17 INFO app started\n2026-07-17 INFO inspection completed",
        }

    def check_vlm_status(self) -> OllamaStatus:
        self.vlm_calls += 1
        return OllamaStatus(
            state="연결됨",
            detail="qwen2.5vl:3b is installed",
            host="http://127.0.0.1:11434",
            model_name="qwen2.5vl:3b",
        )


def _app(monkeypatch: pytest.MonkeyPatch) -> QApplication:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication(sys.argv)


def test_status_view_renders_primary_status_cards_and_logs(monkeypatch) -> None:
    _app(monkeypatch)
    view = StatusView(FakeStatusViewModel())  # type: ignore[arg-type]

    assert view.yolo_card.state_label.text() == "Ready"
    assert view.vlm_card.state_label.text() != "Configured for local Ollama"
    assert view.database_card.detail_label.text() == "database/inspection_results.sqlite3"
    assert view.gpu_card.state_label.text() == "NVIDIA driver visible"
    assert view.log_card.path_label.text() == "logs/error.log"
    assert "inspection completed" in view.log_card.log_text.toPlainText()
    view.close()


def test_status_refresh_button_reloads_status(monkeypatch) -> None:
    _app(monkeypatch)
    viewmodel = FakeStatusViewModel()
    view = StatusView(viewmodel)  # type: ignore[arg-type]
    call_count = viewmodel.calls

    view.refresh_button.click()

    assert viewmodel.calls == call_count + 1
    assert view.vlm_card.state_label.text() != "Configured for local Ollama"
    view.close()


def test_status_viewmodel_reads_recent_log_lines(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "error.log"
    log_path.write_text("\n".join(f"line {index}" for index in range(100)), encoding="utf-8")
    monkeypatch.setattr(settings, "ERROR_LOG_PATH", log_path)
    monkeypatch.setattr(settings, "LOG_DIR", tmp_path)

    logs = StatusViewModel().get_recent_logs(max_lines=3)

    assert logs == "line 97\nline 98\nline 99"


def test_status_viewmodel_handles_missing_log_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "ERROR_LOG_PATH", tmp_path / "missing.log")
    monkeypatch.setattr(settings, "LOG_DIR", tmp_path)

    logs = StatusViewModel().get_recent_logs()

    assert logs == "아직 기록된 실행 로그가 없습니다."
