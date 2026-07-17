from __future__ import annotations

from dataclasses import dataclass
import json
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vlm.vlm_client import VlmClient


@dataclass(frozen=True, slots=True)
class OllamaStatus:
    state: str
    detail: str
    host: str
    model_name: str


class OllamaStatusService:
    def __init__(
        self,
        *,
        host: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        client = VlmClient()
        self.host = (host or client.host).rstrip("/")
        self.model_name = model_name or client.model_name
        self.timeout_seconds = timeout_seconds

    def check_status(self) -> OllamaStatus:
        request = Request(
            url=f"{self.host}/api/tags",
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status_code = int(getattr(response, "status", 200))
                body = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            return OllamaStatus(
                state="응답 오류",
                detail=f"HTTP {exc.code}: {exc.reason}",
                host=self.host,
                model_name=self.model_name,
            )
        except (TimeoutError, socket.timeout) as exc:
            return OllamaStatus(
                state="연결 실패",
                detail=f"Timeout after {self.timeout_seconds:g}s: {exc}",
                host=self.host,
                model_name=self.model_name,
            )
        except URLError as exc:
            return OllamaStatus(
                state="연결 실패",
                detail=str(getattr(exc, "reason", exc)),
                host=self.host,
                model_name=self.model_name,
            )
        except OSError as exc:
            return OllamaStatus(
                state="연결 실패",
                detail=str(exc),
                host=self.host,
                model_name=self.model_name,
            )

        if status_code < 200 or status_code >= 300:
            return OllamaStatus(
                state="응답 오류",
                detail=f"HTTP {status_code}",
                host=self.host,
                model_name=self.model_name,
            )

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            return OllamaStatus(
                state="응답 오류",
                detail=f"Invalid JSON: {exc}",
                host=self.host,
                model_name=self.model_name,
            )

        model_names = _extract_model_names(payload)
        if not model_names:
            return OllamaStatus(
                state="응답 오류",
                detail="No model list in /api/tags response",
                host=self.host,
                model_name=self.model_name,
            )
        if self.model_name not in model_names:
            return OllamaStatus(
                state="모델 없음",
                detail=f"Installed models: {', '.join(sorted(model_names))}",
                host=self.host,
                model_name=self.model_name,
            )
        return OllamaStatus(
            state="연결됨",
            detail=f"{self.model_name} is installed on {self.host}",
            host=self.host,
            model_name=self.model_name,
        )


def _extract_model_names(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    models = payload.get("models")
    if not isinstance(models, list):
        return set()
    names: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        for key in ("name", "model"):
            value = model.get(key)
            if isinstance(value, str) and value:
                names.add(value)
    return names
