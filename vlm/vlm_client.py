from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


class VlmClient:
    """Small Ollama client wrapper for local vision-language inference."""

    def __init__(
        self,
        model_name: str = "qwen2.5vl:3b",
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.1,
        num_ctx: int = 8192,
        num_predict: int = 512,
    ) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def generate(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        image_path: str | Path | None = None,
        image_bytes_list: Sequence[bytes] | None = None,
        image_paths: Sequence[str | Path] | None = None,
    ) -> str:
        """Send a prompt and one or more images to Ollama and return text."""
        images = self._collect_images(
            image_bytes=image_bytes,
            image_path=image_path,
            image_bytes_list=image_bytes_list,
            image_paths=image_paths,
        )

        message: dict[str, Any] = {
            "role": "user",
            "content": prompt,
            "images": images,
        }

        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                "ollama Python package is not installed. "
                "Run: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
            ) from exc

        client = ollama.Client(host=self.host)

        try:
            response = client.chat(
                model=self.model_name,
                messages=[message],
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                },
            )
        except Exception as exc:
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

        if isinstance(response, dict):
            response_message = response.get("message", {})
            content = (
                response_message.get("content", "")
                if isinstance(response_message, dict)
                else ""
            )
        else:
            response_message = getattr(response, "message", None)
            content = (
                getattr(response_message, "content", "")
                if response_message is not None
                else ""
            )

        content_text = str(content).strip()

        if not content_text:
            raise RuntimeError(
                f"Ollama returned an empty VLM response. " f"Model: {self.model_name}"
            )

        return content_text

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

    def _resolve_image_path(self, image_path: str | Path) -> str:
        resolved_image_path = Path(image_path).resolve()

        if not resolved_image_path.is_file():
            raise FileNotFoundError(f"VLM input image not found: {resolved_image_path}")

        return str(resolved_image_path)
