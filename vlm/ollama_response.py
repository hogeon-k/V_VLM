from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OllamaResponseMetadata:
    """Safe Ollama response diagnostics. Duration values use Ollama's original units."""

    http_status: int | None = None
    done: bool | None = None
    done_reason: str | None = None
    content_length: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_duration: int | None = None
    eval_duration: int | None = None
    response_source: str = "none"


class OllamaContentError(ValueError):
    """Raised when an Ollama response has no assistant content but has metadata."""

    def __init__(self, message: str, metadata: OllamaResponseMetadata) -> None:
        super().__init__(message)
        self.metadata = metadata


def normalize_ollama_response(response: Any) -> dict[str, Any]:
    """Convert Ollama SDK response objects into a plain dictionary."""
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(response, "dict"):
        dumped = response.dict()
        if isinstance(dumped, dict):
            return dumped

    result: dict[str, Any] = {}
    for key in (
        "model",
        "created_at",
        "message",
        "response",
        "done",
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "eval_count",
        "prompt_eval_duration",
        "eval_duration",
    ):
        if hasattr(response, key):
            result[key] = _normalize_value(getattr(response, key))
    return result


def build_ollama_metadata(
    response_data: Mapping[str, Any],
    http_status: int | None = None,
) -> OllamaResponseMetadata:
    """Build metadata without requiring assistant content to be present."""
    source, content_length = _content_source_and_length(response_data)
    return OllamaResponseMetadata(
        http_status=http_status,
        done=_optional_bool(response_data.get("done")),
        done_reason=_optional_str(response_data.get("done_reason")),
        content_length=content_length,
        prompt_eval_count=_optional_int(response_data.get("prompt_eval_count")),
        eval_count=_optional_int(response_data.get("eval_count")),
        total_duration=_optional_int(response_data.get("total_duration")),
        load_duration=_optional_int(response_data.get("load_duration")),
        prompt_eval_duration=_optional_int(response_data.get("prompt_eval_duration")),
        eval_duration=_optional_int(response_data.get("eval_duration")),
        response_source=source,
    )


def extract_ollama_content(
    response_data: Mapping[str, Any],
    metadata: OllamaResponseMetadata | None = None,
) -> str:
    """Extract assistant content from /api/chat or /api/generate responses."""
    metadata = metadata or build_ollama_metadata(response_data)
    error = response_data.get("error")
    if error:
        raise RuntimeError(f"Ollama returned an error: {error}")

    message = response_data.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

    response_text = response_data.get("response")
    if isinstance(response_text, str) and response_text.strip():
        return response_text

    raise OllamaContentError(
        "Ollama response JSON did not contain assistant content.",
        metadata,
    )


def build_ollama_debug_lines(response_data: Mapping[str, Any]) -> list[str]:
    """Return safe response-shape diagnostics without image or payload data."""
    message = response_data.get("message")
    message_exists = isinstance(message, Mapping)
    content = message.get("content") if message_exists else None
    content_exists = isinstance(content, str)
    response_text = response_data.get("response")
    lines = [
        f"[DEBUG] Ollama response keys: {sorted(response_data.keys())}",
        f"[DEBUG] message exists: {str(message_exists).lower()}",
        f"[DEBUG] message.content exists: {str(content_exists).lower()}",
        f"[DEBUG] content length: {len(content) if isinstance(content, str) else 0}",
        f"[DEBUG] response field exists: {str(isinstance(response_text, str)).lower()}",
        f"[DEBUG] response field length: {len(response_text) if isinstance(response_text, str) else 0}",
        f"[DEBUG] done: {response_data.get('done')}",
    ]
    for key in (
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "eval_count",
        "prompt_eval_duration",
        "eval_duration",
    ):
        if key in response_data:
            lines.append(f"[DEBUG] {key}: {response_data.get(key)}")
    return lines


def _content_source_and_length(response_data: Mapping[str, Any]) -> tuple[str, int]:
    message = response_data.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return ("message.content" if content.strip() else "none", len(content))

    response_text = response_data.get("response")
    if isinstance(response_text, str):
        return ("response" if response_text.strip() else "none", len(response_text))

    return "none", 0


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return {key: _normalize_value(item) for key, item in dumped.items()}
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, dict):
            return {key: _normalize_value(item) for key, item in dumped.items()}
    if hasattr(value, "__dict__"):
        return {
            key: _normalize_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value
