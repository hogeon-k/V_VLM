from __future__ import annotations

import base64

import pytest

from vlm.ollama_response import OllamaContentError
from vlm.response_schema import VLM_RESPONSE_SCHEMA
from vlm.vlm_client import VlmClient


def install_fake_post(monkeypatch: pytest.MonkeyPatch, response_data: dict[str, object]) -> dict[str, object]:
    calls: dict[str, object] = {}

    def fake_post(self: VlmClient, payload: dict[str, object]) -> tuple[dict[str, object], int]:
        calls["payload"] = payload
        return response_data, 200

    monkeypatch.setattr(VlmClient, "_post_chat", fake_post)
    return calls


def test_vlm_client_defaults() -> None:
    client = VlmClient()

    assert client.host == "http://127.0.0.1:11434"
    assert client.num_ctx == 8192
    assert client.num_predict == 256
    assert client.temperature == 0.0
    assert client.top_p == 0.8
    assert client.top_k == 20
    assert client.repeat_penalty == 1.1
    assert client.seed == 42


def test_vlm_client_http_payload_contains_stream_format_options_and_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_fake_post(monkeypatch, {"message": {"content": "ok"}, "done": True})

    client = VlmClient(
        temperature=0.2,
        top_p=0.7,
        top_k=10,
        repeat_penalty=1.2,
        seed=7,
        num_ctx=9000,
        num_predict=300,
    )
    result = client.generate("prompt", image_bytes=b"image")

    assert result == "ok"
    assert client.last_response_metadata is not None
    assert client.last_response_metadata.response_source == "message.content"
    assert client.last_response_metadata.endpoint == "/api/chat"
    assert client.last_response_metadata.stream is False
    assert client.last_response_metadata.content_length == 2
    assert client.last_response_metadata.done is True
    payload = calls["payload"]
    assert payload["model"] == "qwen2.5vl:3b"
    assert payload["stream"] is False
    assert payload["format"] == VLM_RESPONSE_SCHEMA
    assert payload["options"] == {
        "num_ctx": 9000,
        "num_predict": 300,
        "temperature": 0.2,
        "top_p": 0.7,
        "top_k": 10,
        "repeat_penalty": 1.2,
        "seed": 7,
    }
    assert payload["messages"] == [
        {
            "role": "user",
            "content": "prompt",
            "images": [base64.b64encode(b"image").decode("ascii")],
        }
    ]


def test_vlm_client_passes_image_bytes_list_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_fake_post(monkeypatch, {"message": {"content": "ok"}, "done": True})

    result = VlmClient().generate("prompt", image_bytes_list=[b"full", b"montage"])

    assert result == "ok"
    payload = calls["payload"]
    assert payload["messages"][0]["images"] == [
        base64.b64encode(b"full").decode("ascii"),
        base64.b64encode(b"montage").decode("ascii"),
    ]


def test_vlm_response_schema_describes_visual_feature_quality_hint() -> None:
    visual_feature_schema = VLM_RESPONSE_SCHEMA["properties"]["detections"]["items"][
        "properties"
    ]["visual_feature"]

    assert visual_feature_schema["minLength"] == 1
    assert "YOLO 클래스명만 쓰지 말 것" in visual_feature_schema["description"]


def test_vlm_client_context_error_is_not_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(self: VlmClient, payload: dict[str, object]) -> tuple[dict[str, object], int]:
        raise ValueError("request (4425 tokens) exceeds the available context size (4096 tokens)")

    monkeypatch.setattr(VlmClient, "_post_chat", fake_post)

    with pytest.raises(RuntimeError) as exc_info:
        VlmClient(num_ctx=8192).generate("prompt", image_bytes=b"image")

    message = str(exc_info.value)
    assert "exceeded the configured context size" in message
    assert "num_ctx: 8192" in message
    assert "Increase --vlm-num-ctx or reduce the image size" in message
    assert "ValueError: request (4425 tokens)" in message
    assert "Failed to connect" not in message


def test_vlm_client_model_missing_includes_pull_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(self: VlmClient, payload: dict[str, object]) -> tuple[dict[str, object], int]:
        raise RuntimeError("model qwen2.5vl:3b not found, try pulling it first")

    monkeypatch.setattr(VlmClient, "_post_chat", fake_post)

    with pytest.raises(RuntimeError) as exc_info:
        VlmClient().generate("prompt", image_bytes=b"image")

    message = str(exc_info.value)
    assert "Ollama model is not ready" in message
    assert "ollama pull qwen2.5vl:3b" in message
    assert "RuntimeError: model qwen2.5vl:3b not found" in message


def test_vlm_client_requires_an_image() -> None:
    with pytest.raises(ValueError, match="image_path or image_bytes is required"):
        VlmClient().generate("prompt")


def test_vlm_client_rejects_empty_image_bytes_list() -> None:
    with pytest.raises(ValueError, match="image_bytes_list is empty"):
        VlmClient().generate("prompt", image_bytes_list=[])


def test_vlm_client_rejects_empty_image_bytes_list_item() -> None:
    with pytest.raises(ValueError, match="image_bytes_list item 2 is empty"):
        VlmClient().generate("prompt", image_bytes_list=[b"image", b""])


def test_vlm_client_rejects_missing_image_path(tmp_path) -> None:
    missing_image = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError, match="VLM input image not found"):
        VlmClient().generate("prompt", image_path=missing_image)


def test_vlm_client_rejects_empty_image_bytes() -> None:
    with pytest.raises(ValueError, match="image_bytes is empty"):
        VlmClient().generate("prompt", image_bytes=b"")


def test_vlm_client_extracts_generate_response_field(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_post(monkeypatch, {"response": "generated", "done": True})

    client = VlmClient()

    assert client.generate("prompt", image_bytes=b"image") == "generated"
    assert client.last_response_metadata is not None
    assert client.last_response_metadata.response_source == "response"
    assert client.last_response_metadata.content_length == len("generated")


def test_vlm_client_records_chat_metadata_for_done_true_response(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_post(
        monkeypatch,
        {
            "message": {"role": "assistant", "content": '{"summary":"ok"}'},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 3,
            "eval_count": 4,
            "total_duration": 5,
        },
    )

    client = VlmClient()

    assert client.generate("prompt", image_bytes=b"image") == '{"summary":"ok"}'
    assert client.last_response_metadata is not None
    assert client.last_response_metadata.endpoint == "/api/chat"
    assert client.last_response_metadata.stream is False
    assert client.last_response_metadata.done is True
    assert client.last_response_metadata.done_reason == "stop"
    assert client.last_response_metadata.prompt_eval_count == 3
    assert client.last_response_metadata.eval_count == 4
    assert client.last_response_metadata.total_duration == 5


def test_vlm_client_raises_ollama_error_field(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_post(monkeypatch, {"error": "model not found"})

    client = VlmClient()
    with pytest.raises(RuntimeError, match="Ollama returned an error: model not found"):
        client.generate("prompt", image_bytes=b"image")
    assert client.last_response_metadata is not None
    assert client.last_response_metadata.http_status == 200


def test_vlm_client_raises_empty_assistant_content(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_post(monkeypatch, {"message": {"content": "  "}, "done": True})

    client = VlmClient()
    with pytest.raises(OllamaContentError, match="Ollama response JSON did not contain assistant content"):
        client.generate("prompt", image_bytes=b"image")
    assert client.last_response_metadata is not None
    assert client.last_response_metadata.done is True
    assert client.last_response_metadata.content_length == 2
    assert client.last_response_metadata.response_source == "none"


def test_vlm_client_raises_missing_content_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_post(monkeypatch, {"done": True})

    with pytest.raises(OllamaContentError, match="Ollama response JSON did not contain assistant content"):
        VlmClient().generate("prompt", image_bytes=b"image")


def test_vlm_client_preserves_done_false_empty_content_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_post(
        monkeypatch,
        {
            "message": {"content": ""},
            "done": False,
            "done_reason": None,
            "prompt_eval_count": 0,
            "eval_count": 0,
            "total_duration": 0,
            "load_duration": 12,
            "prompt_eval_duration": 0,
            "eval_duration": 0,
        },
    )

    client = VlmClient()
    with pytest.raises(OllamaContentError) as exc_info:
        client.generate("prompt", image_bytes=b"image")

    metadata = exc_info.value.metadata
    assert metadata.http_status == 200
    assert metadata.done is False
    assert metadata.content_length == 0
    assert metadata.prompt_eval_count == 0
    assert metadata.eval_count == 0
    assert metadata.total_duration == 0
    assert metadata.load_duration == 12
    assert metadata.prompt_eval_duration == 0
    assert metadata.eval_duration == 0
    assert metadata.response_source == "none"
