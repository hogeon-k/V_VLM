from __future__ import annotations

import socket
from urllib.error import HTTPError, URLError

from service import ollama_status_service as module
from service.ollama_status_service import OllamaStatusService


class FakeResponse:
    def __init__(self, body: str, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body.encode("utf-8")


def test_ollama_status_connected_when_model_installed(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "urlopen",
        lambda _request, timeout: FakeResponse('{"models":[{"name":"qwen2.5vl:3b"}]}'),
    )

    status = OllamaStatusService(host="http://127.0.0.1:11434", model_name="qwen2.5vl:3b").check_status()

    assert status.state == "연결됨"


def test_ollama_status_model_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "urlopen",
        lambda _request, timeout: FakeResponse('{"models":[{"name":"llama3"}]}'),
    )

    status = OllamaStatusService(model_name="qwen2.5vl:3b").check_status()

    assert status.state == "모델 없음"
    assert "llama3" in status.detail


def test_ollama_status_connection_failure(monkeypatch) -> None:
    def raise_error(_request: object, timeout: float) -> object:
        raise URLError(ConnectionRefusedError("connection refused"))

    monkeypatch.setattr(module, "urlopen", raise_error)

    status = OllamaStatusService().check_status()

    assert status.state == "연결 실패"


def test_ollama_status_timeout(monkeypatch) -> None:
    def raise_error(_request: object, timeout: float) -> object:
        raise socket.timeout("timed out")

    monkeypatch.setattr(module, "urlopen", raise_error)

    status = OllamaStatusService(timeout_seconds=1.0).check_status()

    assert status.state == "연결 실패"
    assert "Timeout" in status.detail


def test_ollama_status_response_error_for_http_error(monkeypatch) -> None:
    def raise_error(_request: object, timeout: float) -> object:
        raise HTTPError("http://test/api/tags", 500, "server error", hdrs=None, fp=None)

    monkeypatch.setattr(module, "urlopen", raise_error)

    status = OllamaStatusService().check_status()

    assert status.state == "응답 오류"


def test_ollama_status_response_error_for_bad_json(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "urlopen",
        lambda _request, timeout: FakeResponse("not json"),
    )

    status = OllamaStatusService().check_status()

    assert status.state == "응답 오류"
