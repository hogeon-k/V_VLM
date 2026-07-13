from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from vlm.vlm_client import VlmClient


def install_fake_ollama(monkeypatch: pytest.MonkeyPatch, fake_client: object) -> None:
    fake_module = SimpleNamespace(Client=lambda host: fake_client)
    monkeypatch.setitem(sys.modules, "ollama", fake_module)


def test_vlm_client_defaults() -> None:
    client = VlmClient()

    assert client.host == "http://127.0.0.1:11434"
    assert client.num_ctx == 8192
    assert client.num_predict == 512


def test_vlm_client_passes_ollama_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeClient:
        def chat(self, **kwargs: object) -> dict[str, object]:
            calls.update(kwargs)
            return {"message": {"content": "ok"}}

    install_fake_ollama(monkeypatch, FakeClient())

    client = VlmClient(temperature=0.2, num_ctx=9000, num_predict=300)
    result = client.generate("prompt", image_bytes=b"image")

    assert result == "ok"
    assert calls["options"] == {
        "temperature": 0.2,
        "num_ctx": 9000,
        "num_predict": 300,
    }
    assert calls["messages"] == [
        {"role": "user", "content": "prompt", "images": [b"image"]}
    ]


def test_vlm_client_context_error_is_not_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def chat(self, **kwargs: object) -> dict[str, object]:
            raise ValueError("request (4425 tokens) exceeds the available context size (4096 tokens)")

    install_fake_ollama(monkeypatch, FakeClient())

    with pytest.raises(RuntimeError) as exc_info:
        VlmClient(num_ctx=8192).generate("prompt", image_bytes=b"image")

    message = str(exc_info.value)
    assert "exceeded the configured context size" in message
    assert "num_ctx: 8192" in message
    assert "Increase --vlm-num-ctx or reduce the image size" in message
    assert "ValueError: request (4425 tokens)" in message
    assert "Failed to connect" not in message


def test_vlm_client_model_missing_includes_pull_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def chat(self, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("model qwen2.5vl:3b not found, try pulling it first")

    install_fake_ollama(monkeypatch, FakeClient())

    with pytest.raises(RuntimeError) as exc_info:
        VlmClient().generate("prompt", image_bytes=b"image")

    message = str(exc_info.value)
    assert "Ollama model is not ready" in message
    assert "ollama pull qwen2.5vl:3b" in message
    assert "RuntimeError: model qwen2.5vl:3b not found" in message


def test_vlm_client_requires_an_image() -> None:
    with pytest.raises(ValueError, match="image_path or image_bytes is required"):
        VlmClient().generate("prompt")


def test_vlm_client_rejects_missing_image_path(tmp_path) -> None:
    missing_image = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError, match="VLM input image not found"):
        VlmClient().generate("prompt", image_path=missing_image)


def test_vlm_client_rejects_empty_image_bytes() -> None:
    with pytest.raises(ValueError, match="image_bytes is empty"):
        VlmClient().generate("prompt", image_bytes=b"")


def test_vlm_client_rejects_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def chat(self, **kwargs: object) -> dict[str, object]:
            return {"message": {"content": "  "}}

    install_fake_ollama(monkeypatch, FakeClient())

    with pytest.raises(RuntimeError, match="empty VLM response"):
        VlmClient().generate("prompt", image_bytes=b"image")
