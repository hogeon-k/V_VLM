from __future__ import annotations

import json
import queue
import subprocess
import threading
from pathlib import Path
from typing import Callable

import pytest

from service import tensorrt_persistent_worker as worker_module
from service.tensorrt_persistent_worker import (
    TensorRtPersistentWorker,
    TensorRtWorkerProtocolError,
    TensorRtWorkerRemoteError,
    TensorRtWorkerStartupError,
    TensorRtWorkerTimeoutError,
)


class QueueStream:
    def __init__(self, lines: list[bytes] | None = None) -> None:
        self.items: queue.Queue[bytes] = queue.Queue()
        for line in lines or []:
            self.feed(line)

    def feed(self, line: bytes) -> None:
        self.items.put(line if line.endswith(b"\n") else line + b"\n")

    def readline(self) -> bytes:
        return self.items.get(timeout=5)

    def close(self) -> None:
        self.items.put(b"")


class RequestSink:
    def __init__(self, responder: Callable[[dict[str, object]], bytes | None]) -> None:
        self.responder = responder
        self.writes: list[bytes] = []
        self.stdout: QueueStream | None = None

    def write(self, payload: bytes) -> int:
        self.writes.append(payload)
        request = json.loads(payload.decode("utf-8"))
        response = self.responder(request)
        if response is not None:
            assert self.stdout is not None
            self.stdout.feed(response)
        return len(payload)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeProcess:
    next_pid = 4100

    def __init__(
        self,
        ready: bytes,
        responder: Callable[[dict[str, object]], bytes | None],
        stderr: list[bytes] | None = None,
    ) -> None:
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.stdout = QueueStream([ready])
        self.stderr = QueueStream(stderr)
        self.stdin = RequestSink(responder)
        self.stdin.stdout = self.stdout
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self.stdout.close()
        self.stderr.close()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.stdout.close()
        self.stderr.close()


def ready(ok: bool = True) -> bytes:
    return json.dumps(
        {
            "event": "ready",
            "ok": ok,
            "backend": "tensorrt",
            "engine_label": "fp16",
            "device_id": 0,
            "startup_ms": 42.5,
            "message": "engine failed" if not ok else None,
        }
    ).encode("utf-8")


def successful_responder(request: dict[str, object]) -> bytes:
    if request["command"] == "shutdown":
        return json.dumps(
            {
                "request_id": request["request_id"],
                "ok": True,
                "status": "shutdown",
            }
        ).encode("utf-8")
    return json.dumps(
        {
            "request_id": request["request_id"],
            "ok": True,
            "backend": "tensorrt",
            "engine_label": "fp16",
            "detections": [],
            "timing_ms": {"total": 3.2},
        }
    ).encode("utf-8")


def make_worker(tmp_path: Path, **kwargs: object) -> TensorRtPersistentWorker:
    options: dict[str, object] = {
        "startup_timeout_seconds": 0.2,
        "inference_timeout_seconds": 0.2,
        "shutdown_timeout_seconds": 0.2,
    }
    options.update(kwargs)
    return TensorRtPersistentWorker(
        executable_path=tmp_path / "pcb_onnx_infer.exe",
        engine_path=tmp_path / "best.engine",
        metadata_path=tmp_path / "metadata.json",
        engine_label="fp16",
        device_id=0,
        **options,
    )


def install_process(
    monkeypatch: pytest.MonkeyPatch,
    process: FakeProcess,
) -> None:
    monkeypatch.setattr(worker_module.subprocess, "Popen", lambda *args, **kwargs: process)


def test_worker_command_contains_persistent_arguments(tmp_path) -> None:
    worker = make_worker(tmp_path)

    command = worker.build_command()

    assert command[-1] == "--worker"
    assert command[command.index("--backend") + 1] == "tensorrt"
    assert command[command.index("--engine-label") + 1] == "fp16"
    assert "--image" not in command


def test_utf8_json_request_preserves_unicode_path(monkeypatch, tmp_path) -> None:
    process = FakeProcess(ready(), successful_responder)
    install_process(monkeypatch, process)
    worker = make_worker(tmp_path)
    image = tmp_path / "한글 폴더" / "검사.JPG"

    worker.infer(image, 0.15, 0.5)

    request = json.loads(process.stdin.writes[0].decode("utf-8"))
    assert request["image"] == str(image)
    assert b"\\ud55c" not in process.stdin.writes[0]
    worker.stop()


def test_ready_handshake_and_infer_success(monkeypatch, tmp_path) -> None:
    process = FakeProcess(ready(), successful_responder)
    install_process(monkeypatch, process)
    worker = make_worker(tmp_path)

    result = worker.infer(tmp_path / "image.jpg", 0.15, 0.5)

    assert result["ok"] is True
    assert result["ipc_roundtrip_ms"] >= 0
    assert worker.startup_ms == pytest.approx(42.5)
    assert worker.pid == process.pid
    worker.stop()


def test_ready_failure_raises_and_terminates(monkeypatch, tmp_path) -> None:
    process = FakeProcess(ready(False), successful_responder)
    install_process(monkeypatch, process)
    worker = make_worker(tmp_path)

    with pytest.raises(TensorRtWorkerStartupError, match="engine failed"):
        worker.start()

    assert process.terminated is True
    assert worker.is_alive() is False


def test_remote_infer_error_does_not_kill_worker(monkeypatch, tmp_path) -> None:
    def responder(request: dict[str, object]) -> bytes:
        return json.dumps(
            {
                "request_id": request["request_id"],
                "ok": False,
                "error_type": "ImageLoadError",
                "message": "decode failed",
            }
        ).encode("utf-8")

    process = FakeProcess(ready(), responder)
    install_process(monkeypatch, process)
    worker = make_worker(tmp_path)

    with pytest.raises(TensorRtWorkerRemoteError, match="decode failed"):
        worker.infer(tmp_path / "bad.jpg", 0.15, 0.5)

    assert worker.is_alive() is True
    worker.stop()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (b"not-json", "invalid JSON"),
        (b"\xff\xfe", "non-UTF-8"),
    ],
)
def test_invalid_protocol_output_terminates_worker(
    monkeypatch,
    tmp_path,
    response: bytes,
    message: str,
) -> None:
    process = FakeProcess(ready(), lambda request: response)
    install_process(monkeypatch, process)
    worker = make_worker(tmp_path)

    with pytest.raises(TensorRtWorkerProtocolError, match=message):
        worker.infer(tmp_path / "image.jpg", 0.15, 0.5)

    assert process.terminated is True


def test_request_id_mismatch_terminates_worker(monkeypatch, tmp_path) -> None:
    response = json.dumps(
        {"request_id": "wrong", "ok": True, "detections": []}
    ).encode("utf-8")
    process = FakeProcess(ready(), lambda request: response)
    install_process(monkeypatch, process)
    worker = make_worker(tmp_path)

    with pytest.raises(TensorRtWorkerProtocolError, match="request_id mismatch"):
        worker.infer(tmp_path / "image.jpg", 0.15, 0.5)

    assert process.terminated is True


def test_inference_timeout_terminates_worker(monkeypatch, tmp_path) -> None:
    process = FakeProcess(ready(), lambda request: None)
    install_process(monkeypatch, process)
    worker = make_worker(tmp_path, inference_timeout_seconds=0.01)

    with pytest.raises(TensorRtWorkerTimeoutError, match="timed out"):
        worker.infer(tmp_path / "image.jpg", 0.15, 0.5)

    assert process.terminated is True


def test_worker_restarts_after_timeout(monkeypatch, tmp_path) -> None:
    first = FakeProcess(ready(), lambda request: None)
    second = FakeProcess(ready(), successful_responder)
    processes = [first, second]
    monkeypatch.setattr(
        worker_module.subprocess,
        "Popen",
        lambda *args, **kwargs: processes.pop(0),
    )
    worker = make_worker(tmp_path, inference_timeout_seconds=0.01)

    with pytest.raises(TensorRtWorkerTimeoutError):
        worker.infer(tmp_path / "first.jpg", 0.15, 0.5)
    response = worker.infer(tmp_path / "second.jpg", 0.15, 0.5)

    assert response["ok"] is True
    assert worker.pid == second.pid
    worker.stop()


def test_startup_timeout_terminates_worker(monkeypatch, tmp_path) -> None:
    process = FakeProcess(ready(), successful_responder)
    process.stdout = QueueStream()
    process.stdin.stdout = process.stdout
    install_process(monkeypatch, process)
    worker = make_worker(tmp_path, startup_timeout_seconds=0.01)

    with pytest.raises(TensorRtWorkerTimeoutError, match="startup timed out"):
        worker.start()

    assert process.terminated is True


def test_terminate_timeout_escalates_to_kill(monkeypatch, tmp_path) -> None:
    class StubbornProcess(FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            if self.terminated and not self.killed:
                raise subprocess.TimeoutExpired("worker", timeout)
            self.returncode = -9 if self.killed else 0
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.stdout.close()
            self.stderr.close()

    process = StubbornProcess(ready(), lambda request: None)
    install_process(monkeypatch, process)
    worker = make_worker(tmp_path, inference_timeout_seconds=0.01)

    with pytest.raises(TensorRtWorkerTimeoutError):
        worker.infer(tmp_path / "image.jpg", 0.15, 0.5)

    assert process.terminated is True
    assert process.killed is True


def test_stdout_eof_reports_crash_and_stderr(monkeypatch, tmp_path) -> None:
    process = FakeProcess(ready(), lambda request: None, stderr=[b"CUDA crashed"])
    install_process(monkeypatch, process)
    worker = make_worker(tmp_path)
    worker.start()
    process.stdout.close()

    with pytest.raises(TensorRtWorkerProtocolError, match="stdout closed"):
        worker.infer(tmp_path / "image.jpg", 0.15, 0.5)


def test_graceful_shutdown_is_idempotent(monkeypatch, tmp_path) -> None:
    process = FakeProcess(ready(), successful_responder)
    install_process(monkeypatch, process)
    worker = make_worker(tmp_path)
    worker.start()

    worker.stop()
    worker.stop()

    request = json.loads(process.stdin.writes[0].decode("utf-8"))
    assert request["command"] == "shutdown"
    assert worker.is_alive() is False


def test_stderr_is_drained_and_truncated(monkeypatch, tmp_path) -> None:
    process = FakeProcess(
        ready(),
        successful_responder,
        stderr=[b"line one", "경고".encode("cp949")],
    )
    install_process(monkeypatch, process)
    worker = make_worker(tmp_path)
    worker.start()

    assert "line one" in worker.stderr_excerpt()
    assert "경고" in worker.stderr_excerpt()
    worker.stop()


def test_concurrent_infer_requests_are_serialized(monkeypatch, tmp_path) -> None:
    process = FakeProcess(ready(), successful_responder)
    install_process(monkeypatch, process)
    worker = make_worker(tmp_path)
    errors: list[Exception] = []

    def invoke(index: int) -> None:
        try:
            worker.infer(tmp_path / f"{index}.jpg", 0.15, 0.5)
        except Exception as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)

    threads = [threading.Thread(target=invoke, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    infer_requests = [
        json.loads(item.decode("utf-8"))
        for item in process.stdin.writes
        if json.loads(item.decode("utf-8"))["command"] == "infer"
    ]
    assert len(infer_requests) == 4
    assert len({item["request_id"] for item in infer_requests}) == 4
    worker.stop()
