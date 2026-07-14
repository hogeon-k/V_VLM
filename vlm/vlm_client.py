from __future__ import annotations

import base64
import json
import socket
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vlm.ollama_response import (
    OllamaResponseMetadata,
    build_ollama_metadata,
    build_ollama_debug_lines,
    extract_ollama_content,
    normalize_ollama_response,
)
from vlm.response_schema import VLM_RESPONSE_SCHEMA


class VlmClient:
    """Small Ollama client wrapper for local vision-language inference."""

    def __init__(
        self,
        model_name: str = "qwen2.5vl:3b",
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        top_p: float = 0.8,
        top_k: int = 20,
        repeat_penalty: float = 1.1,
        seed: int = 42,
        num_ctx: int = 8192,
        num_predict: int = 256,
        response_schema: dict[str, Any] | None = None,
        debug_response: bool = False,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.seed = seed
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.response_schema = response_schema or VLM_RESPONSE_SCHEMA
        self.debug_response = debug_response
        self.timeout_seconds = timeout_seconds
        self.last_response_metadata: OllamaResponseMetadata | None = None
        self.last_response_data: dict[str, Any] | None = None
        self.last_request_summary: dict[str, Any] | None = None
        self.last_error_type: str = ""
        self.last_error_message: str = ""
        self.endpoint = "/api/chat"
        self.stream = False

    def generate(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        image_path: str | Path | None = None,
        image_bytes_list: Sequence[bytes] | None = None,
        image_paths: Sequence[str | Path] | None = None,
    ) -> str:
        """Send a prompt and one or more images to Ollama and return text."""
        self.last_response_metadata = None
        self.last_response_data = None
        self.last_request_summary = None
        self.last_error_type = ""
        self.last_error_message = ""
        images = self._collect_images(
            image_bytes=image_bytes,
            image_path=image_path,
            image_bytes_list=image_bytes_list,
            image_paths=image_paths,
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": self._encode_images_for_http(images),
                }
            ],
            "stream": False,
            "format": self.response_schema,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty,
                "seed": self.seed,
            },
        }
        self.last_request_summary = self._summarize_payload(payload)

        try:
            response_data, status_code = self._post_chat(payload)
        except Exception as exc:
            self.last_response_metadata = None
            if not self.last_error_type:
                self.last_error_type = type(exc).__name__
            self.last_error_message = str(exc)
            error_text = str(exc)
            error_lower = error_text.lower()

            if "not found" in error_lower or "pull" in error_lower:
                raise RuntimeError(
                    f"Ollama model is not ready: {self.model_name}. "
                    f"Run: ollama pull {self.model_name}. "
                    f"Original error: {type(exc).__name__}: {error_text}"
                ) from exc

            if (
                "context size" in error_lower
                or "context length" in error_lower
                or "exceeds the available context" in error_lower
                or "num_ctx" in error_lower
            ):
                raise RuntimeError(
                    f"Ollama VLM request exceeded the configured context size. "
                    f"Host: {self.host}, model: {self.model_name}, num_ctx: {self.num_ctx}. "
                    f"The request token count is larger than the available context. "
                    f"Increase --vlm-num-ctx or reduce the image size passed to the VLM. "
                    f"Original error: {type(exc).__name__}: {error_text}"
                ) from exc

            if (
                "connection refused" in error_lower
                or "failed to establish" in error_lower
                or "connecterror" in error_lower
                or "connection error" in error_lower
                or "connect error" in error_lower
                or "winerror 10061" in error_lower
            ):
                raise RuntimeError(
                    f"Failed to connect to Ollama at {self.host}. "
                    f"Original error: {type(exc).__name__}: {error_text}"
                ) from exc

            raise RuntimeError(
                f"Ollama VLM request failed. "
                f"Host: {self.host}, model: {self.model_name}. "
                f"Original error: {type(exc).__name__}: {error_text}"
            ) from exc

        self.last_error_type = ""
        self.last_error_message = ""
        self.last_response_data = response_data
        metadata = build_ollama_metadata(
            response_data,
            http_status=status_code,
            endpoint=self.endpoint,
            stream=self.stream,
        )
        self.last_response_metadata = metadata

        if self.debug_response:
            print(f"[DEBUG] Ollama status code: {status_code}")
            if self.last_request_summary is not None:
                print(f"[DEBUG] Ollama request JSON bytes: {self.last_request_summary['json_size_bytes']}")
                print(f"[DEBUG] Ollama request image count: {self.last_request_summary['image_count']}")
                print(f"[DEBUG] Ollama request base64 lengths: {self.last_request_summary['base64_lengths']}")
                print(f"[DEBUG] Ollama request prompt length: {self.last_request_summary['prompt_length']}")
            for line in build_ollama_debug_lines(response_data):
                print(line)
        return extract_ollama_content(response_data, metadata)

    def unload_model(self) -> bool:
        """Ask Ollama to unload this model from its runner."""
        payload = {
            "model": self.model_name,
            "keep_alive": 0,
        }
        request = Request(
            url=f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=min(self.timeout_seconds, 30.0)) as response:
                body = response.read().decode("utf-8", errors="replace")
                status_code = int(getattr(response, "status", 200))
        except (HTTPError, URLError, TimeoutError, socket.timeout):
            return False

        if status_code != 200:
            return False
        try:
            response_data = json.loads(body)
        except json.JSONDecodeError:
            return False
        return response_data.get("done_reason") == "unload"

    def _post_chat(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        request = Request(
            url=f"{self.host}{self.endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status_code = int(getattr(response, "status", 200))
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self.last_error_type = "http_error"
            self.last_error_message = body
            raise RuntimeError(
                f"Ollama HTTP request failed. status_code={exc.code}. body={body}"
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            self.last_error_type = "timeout"
            self.last_error_message = str(exc)
            raise RuntimeError(f"Ollama HTTP request timed out after {self.timeout_seconds}s") from exc
        except URLError as exc:
            reason = str(getattr(exc, "reason", exc))
            if "timed out" in reason.lower() or "timeout" in reason.lower():
                self.last_error_type = "timeout"
                self.last_error_message = reason
                raise RuntimeError(f"Ollama HTTP request timed out after {self.timeout_seconds}s") from exc
            self.last_error_type = "connection_error"
            self.last_error_message = str(exc)
            raise RuntimeError(f"Failed to connect to Ollama at {self.host}. {exc}") from exc

        if not body.strip():
            self.last_error_type = "empty_http_body"
            self.last_error_message = f"status_code={status_code}"
            raise RuntimeError(
                "Ollama returned an empty HTTP response body. "
                f"status_code={status_code}"
            )

        try:
            response_data = json.loads(body)
        except json.JSONDecodeError as exc:
            self.last_error_type = "invalid_http_json"
            self.last_error_message = exc.msg
            raise RuntimeError(
                f"Ollama returned invalid JSON. status_code={status_code}. "
                f"Original error: {exc.msg}"
            ) from exc

        return normalize_ollama_response(response_data), status_code

    def _collect_images(
        self,
        image_bytes: bytes | None,
        image_path: str | Path | None,
        image_bytes_list: Sequence[bytes] | None,
        image_paths: Sequence[str | Path] | None,
    ) -> list[bytes | str]:
        images: list[bytes | str] = []

        if image_bytes is not None:
            if not image_bytes:
                raise ValueError("VLM image_bytes is empty.")
            images.append(image_bytes)

        if image_path is not None:
            images.append(self._resolve_image_path(image_path))

        if image_bytes_list is not None:
            if not image_bytes_list:
                raise ValueError("VLM image_bytes_list is empty.")
            for index, item in enumerate(image_bytes_list, start=1):
                if not item:
                    raise ValueError(f"VLM image_bytes_list item {index} is empty.")
                images.append(item)

        if image_paths is not None:
            if not image_paths:
                raise ValueError("VLM image_paths is empty.")
            images.extend(self._resolve_image_path(path) for path in image_paths)

        if not images:
            raise ValueError("VLM image_path or image_bytes is required.")

        return images

    def _encode_images_for_http(self, images: Sequence[bytes | str]) -> list[str]:
        encoded_images: list[str] = []
        for image in images:
            if isinstance(image, bytes):
                image_bytes = image
            else:
                image_bytes = Path(image).read_bytes()
            encoded_images.append(base64.b64encode(image_bytes).decode("ascii"))
        return encoded_images

    def _resolve_image_path(self, image_path: str | Path) -> str:
        resolved_image_path = Path(image_path).resolve()

        if not resolved_image_path.is_file():
            raise FileNotFoundError(f"VLM input image not found: {resolved_image_path}")

        return str(resolved_image_path)

    def _summarize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        message = payload["messages"][0]
        images = message.get("images", [])
        base64_lengths = [len(image) for image in images] if isinstance(images, list) else []
        return {
            "json_size_bytes": len(json.dumps(payload).encode("utf-8")),
            "image_count": len(base64_lengths),
            "base64_lengths": base64_lengths,
            "prompt_length": len(str(message.get("content", ""))),
            "format": "schema" if isinstance(payload.get("format"), dict) else payload.get("format"),
            "has_options": "options" in payload,
        }
