from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

_EOF = object()


class TensorRtWorkerError(RuntimeError):
    """Base error for persistent TensorRT worker failures."""


class TensorRtWorkerStartupError(TensorRtWorkerError):
    """Raised when the worker cannot complete its ready handshake."""


class TensorRtWorkerProtocolError(TensorRtWorkerError):
    """Raised when stdout does not contain the expected UTF-8 JSONL response."""


class TensorRtWorkerTimeoutError(TensorRtWorkerError):
    """Raised when a worker response exceeds its deadline."""


class TensorRtWorkerRemoteError(TensorRtWorkerError):
    """Raised for a non-fatal request error returned by the C++ worker."""

    def __init__(self, error_type: str, message: str) -> None:
        self.error_type = error_type
        super().__init__(f"{error_type}: {message}")


class TensorRtPersistentWorker:
    """Own one TensorRT JSONL worker process and serialize inference requests."""

    def __init__(
        self,
        executable_path: str | Path,
        engine_path: str | Path,
        metadata_path: str | Path,
        engine_label: str,
        device_id: int,
        image_size: int = 960,
        startup_timeout_seconds: float = 120.0,
        inference_timeout_seconds: float = 120.0,
        shutdown_timeout_seconds: float = 5.0,
        stderr_line_limit: int = 100,
    ) -> None:
        self.executable_path = Path(executable_path)
        self.engine_path = Path(engine_path)
        self.metadata_path = Path(metadata_path)
        self.engine_label = engine_label
        self.device_id = int(device_id)
        self.image_size = int(image_size)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.inference_timeout_seconds = float(inference_timeout_seconds)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)

        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_queue: queue.Queue[bytes | object] = queue.Queue()
        self._stderr_lines: deque[bytes] = deque(maxlen=stderr_line_limit)
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self.startup_ms: float | None = None

    @property
    def configuration_key(self) -> tuple[Path, Path, str, int]:
        return (
            self.engine_path.resolve(),
            self.metadata_path.resolve(),
            self.engine_label,
            self.device_id,
        )

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.poll() is None else None

    def build_command(self) -> list[str]:
        return [
            str(self.executable_path),
            "--backend",
            "tensorrt",
            "--engine",
            str(self.engine_path),
            "--metadata",
            str(self.metadata_path),
            "--engine-label",
            self.engine_label,
            "--device-id",
            str(self.device_id),
            "--imgsz",
            str(self.image_size),
            "--worker",
        ]

    def start(self) -> None:
        with self._lock:
            if self.is_alive():
                return
            self._clear_process_state()
            command = self.build_command()
            started = time.perf_counter()
            try:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                    shell=False,
                    bufsize=0,
                )
            except OSError as exc:
                self._clear_process_state()
                raise TensorRtWorkerStartupError(
                    f"Failed to start TensorRT worker: {exc}; command={command}"
                ) from exc

            process = self._process
            assert process.stdout is not None
            assert process.stderr is not None
            self._stdout_thread = threading.Thread(
                target=self._read_stdout,
                args=(process.stdout, self._stdout_queue),
                name="tensorrt-worker-stdout",
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                args=(process.stderr,),
                name="tensorrt-worker-stderr",
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

            try:
                ready = self._read_json_response(
                    timeout=self.startup_timeout_seconds,
                    context="startup",
                )
                if ready.get("event") != "ready":
                    raise TensorRtWorkerProtocolError(
                        f"Expected ready handshake, got: {ready}"
                    )
                if ready.get("ok") is not True:
                    raise TensorRtWorkerStartupError(
                        f"TensorRT worker rejected startup: {ready.get('message', 'unknown error')}"
                    )
                self.startup_ms = float(
                    ready.get("startup_ms", (time.perf_counter() - started) * 1000)
                )
                logger.info(
                    "TensorRT persistent worker ready pid=%s engine=%s label=%s startup_ms=%.3f",
                    self.pid,
                    self.engine_path,
                    self.engine_label,
                    self.startup_ms,
                )
            except Exception:
                self._terminate_process()
                raise

    def infer(
        self,
        image_path: str | Path,
        confidence: float,
        iou: float,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.start()
            request_id = uuid4().hex
            request: dict[str, object] = {
                "request_id": request_id,
                "command": "infer",
                "image": str(Path(image_path)),
                "confidence": float(confidence),
                "iou": float(iou),
            }
            if output_dir is not None:
                request["output"] = str(Path(output_dir))

            started = time.perf_counter()
            try:
                self._write_request(request)
                response = self._read_json_response(
                    timeout=self.inference_timeout_seconds,
                    context=f"infer request_id={request_id}",
                )
            except (TensorRtWorkerTimeoutError, TensorRtWorkerProtocolError):
                self._terminate_process()
                raise

            if response.get("request_id") != request_id:
                self._terminate_process()
                raise TensorRtWorkerProtocolError(
                    f"TensorRT worker request_id mismatch: expected={request_id}; "
                    f"actual={response.get('request_id')}"
                )
            if response.get("ok") is not True:
                raise TensorRtWorkerRemoteError(
                    str(response.get("error_type") or "WorkerError"),
                    str(response.get("message") or "TensorRT worker request failed"),
                )
            response["ipc_roundtrip_ms"] = (time.perf_counter() - started) * 1000
            return response

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                return
            if process.poll() is None:
                request_id = uuid4().hex
                try:
                    self._write_request(
                        {"request_id": request_id, "command": "shutdown"}
                    )
                    response = self._read_json_response(
                        timeout=self.shutdown_timeout_seconds,
                        context="shutdown",
                    )
                    if (
                        response.get("request_id") != request_id
                        or response.get("ok") is not True
                        or response.get("status") != "shutdown"
                    ):
                        raise TensorRtWorkerProtocolError(
                            f"Invalid shutdown response: {response}"
                        )
                    process.wait(timeout=self.shutdown_timeout_seconds)
                except (
                    OSError,
                    subprocess.TimeoutExpired,
                    TensorRtWorkerError,
                ):
                    self._terminate_process()
                else:
                    self._clear_process_state()
            else:
                self._clear_process_state()

    def restart(self) -> None:
        with self._lock:
            self.stop()
            self.start()

    def is_alive(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def stderr_excerpt(self, limit: int = 1200) -> str:
        raw = b"".join(self._stderr_lines)
        text = self._decode_log_output(raw)
        return text if len(text) <= limit else text[-limit:]

    def __enter__(self) -> TensorRtPersistentWorker:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def _write_request(self, request: dict[str, object]) -> None:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise TensorRtWorkerProtocolError(
                f"TensorRT worker is not running. stderr={self.stderr_excerpt()}"
            )
        payload = (
            self._serialize_request(request) + b"\n"
        )
        try:
            process.stdin.write(payload)
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise TensorRtWorkerProtocolError(
                f"Failed to write TensorRT worker request: {exc}; "
                f"stderr={self.stderr_excerpt()}"
            ) from exc

    def _read_json_response(self, timeout: float, context: str) -> dict[str, Any]:
        try:
            item = self._stdout_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TensorRtWorkerTimeoutError(
                f"TensorRT worker {context} timed out after {timeout:g}s; "
                f"stderr={self.stderr_excerpt()}"
            ) from exc
        if item is _EOF:
            returncode = self._process.poll() if self._process is not None else None
            raise TensorRtWorkerProtocolError(
                f"TensorRT worker stdout closed during {context}; "
                f"returncode={returncode}; stderr={self.stderr_excerpt()}"
            )
        assert isinstance(item, bytes)
        try:
            line = item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise TensorRtWorkerProtocolError(
                f"TensorRT worker emitted non-UTF-8 protocol output during {context}: {exc}"
            ) from exc
        try:
            payload = json.loads(line, parse_constant=self._reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise TensorRtWorkerProtocolError(
                f"TensorRT worker emitted invalid JSON during {context}: {exc}; "
                f"line={line[:500]!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise TensorRtWorkerProtocolError(
                f"TensorRT worker response must be a JSON object during {context}."
            )
        return payload

    def _read_stdout(
        self,
        stream: Any,
        response_queue: queue.Queue[bytes | object],
    ) -> None:
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                response_queue.put(line.rstrip(b"\r\n"))
        finally:
            response_queue.put(_EOF)

    def _read_stderr(self, stream: Any) -> None:
        while True:
            line = stream.readline()
            if not line:
                return
            self._stderr_lines.append(line)

    def _terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self.shutdown_timeout_seconds)
        self._clear_process_state()

    def _clear_process_state(self) -> None:
        process = self._process
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        current_thread = threading.current_thread()
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None and thread is not current_thread:
                thread.join(timeout=0.5)
        self._process = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._stdout_queue = queue.Queue()
        self._stderr_lines.clear()

    @staticmethod
    def _decode_log_output(data: bytes) -> str:
        for encoding in ("utf-8", "cp949"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _serialize_request(request: dict[str, object]) -> bytes:
        try:
            return json.dumps(
                request,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise TensorRtWorkerProtocolError(
                f"TensorRT worker request is not valid JSON: {exc}"
            ) from exc

    @staticmethod
    def _reject_json_constant(value: str) -> object:
        raise ValueError(f"non-standard numeric constant {value}")
