from pathlib import Path

from config.detector_settings import DetectorSettings
from service.yolo_service import YoloService
from service.tensorrt_detector_adapter import TensorRtDetectorAdapter
from service.tensorrt_persistent_worker import (
    TensorRtWorkerProtocolError,
    TensorRtWorkerRemoteError,
)

from tests.test_tensorrt_detector_adapter import make_adapter_files
from viewmodel import inspection_viewmodel as viewmodel_module
from viewmodel.inspection_viewmodel import InspectionViewModel


def test_adapter_uses_persistent_payload_without_result_json(monkeypatch, tmp_path) -> None:
    executable, engine, metadata, image = make_adapter_files(tmp_path)
    output_dirs: list[Path] = []

    class FakeWorker:
        startup_ms = 100.0
        pid = 1234

        def infer(self, image_path, confidence, iou, output_dir):
            output_dirs.append(Path(output_dir))
            return {
                "ok": True,
                "backend": "tensorrt",
                "engine_label": "fp16",
                "detections": [],
                "timing_ms": {"total": 4.0},
                "ipc_roundtrip_ms": 5.0,
            }

        def stderr_excerpt(self):
            return ""

        def stop(self):
            return None

    adapter = TensorRtDetectorAdapter(executable, engine, metadata)
    monkeypatch.setattr(adapter, "_get_persistent_worker", lambda: FakeWorker())

    result = adapter.detect(image)

    assert result.detections == []
    assert adapter.last_metadata is not None
    assert adapter.last_metadata.execution_mode == "persistent"
    assert adapter.last_metadata.ipc_roundtrip_ms == 5.0
    assert not output_dirs[0].exists()


def test_adapter_falls_back_to_oneshot_on_protocol_failure(monkeypatch, tmp_path) -> None:
    executable, engine, metadata, image = make_adapter_files(tmp_path)
    adapter = TensorRtDetectorAdapter(executable, engine, metadata)

    class FailedWorker:
        def infer(self, *args, **kwargs):
            raise TensorRtWorkerProtocolError("worker crashed")

    sentinel = object()
    monkeypatch.setattr(adapter, "_get_persistent_worker", lambda: FailedWorker())
    monkeypatch.setattr(adapter, "_infer_oneshot", lambda *args: sentinel)

    assert adapter.detect(image) is sentinel


def test_adapter_does_not_fallback_when_disabled(monkeypatch, tmp_path) -> None:
    executable, engine, metadata, image = make_adapter_files(tmp_path)
    adapter = TensorRtDetectorAdapter(
        executable,
        engine,
        metadata,
        fallback_to_oneshot=False,
    )

    class FailedWorker:
        def infer(self, *args, **kwargs):
            raise TensorRtWorkerProtocolError("worker crashed")

    monkeypatch.setattr(adapter, "_get_persistent_worker", lambda: FailedWorker())

    try:
        adapter.detect(image)
    except Exception as exc:
        assert "persistent worker failed" in str(exc)
    else:
        raise AssertionError("Expected persistent worker failure")


def test_adapter_does_not_retry_remote_request_error(monkeypatch, tmp_path) -> None:
    executable, engine, metadata, image = make_adapter_files(tmp_path)
    adapter = TensorRtDetectorAdapter(executable, engine, metadata)
    called = False

    class FailedWorker:
        def infer(self, *args, **kwargs):
            raise TensorRtWorkerRemoteError("ImageLoadError", "bad image")

    def oneshot(*args):
        nonlocal called
        called = True

    monkeypatch.setattr(adapter, "_get_persistent_worker", lambda: FailedWorker())
    monkeypatch.setattr(adapter, "_infer_oneshot", oneshot)

    try:
        adapter.detect(image)
    except Exception as exc:
        assert "bad image" in str(exc)
    else:
        raise AssertionError("Expected remote worker error")
    assert called is False


def test_viewmodel_reuses_service_until_detector_settings_change(monkeypatch) -> None:
    current = DetectorSettings(detector_backend="tensorrt")
    detectors: list[object] = []

    class FakeDetector:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def load_settings() -> DetectorSettings:
        return current

    def create_service(settings: DetectorSettings) -> YoloService:
        detector = FakeDetector()
        detectors.append(detector)
        return YoloService(detector)

    monkeypatch.setattr(viewmodel_module, "load_detector_settings", load_settings)
    monkeypatch.setattr(
        viewmodel_module,
        "create_yolo_service_from_settings",
        create_service,
    )
    viewmodel = InspectionViewModel()

    viewmodel._apply_detector_settings()
    first_service = viewmodel.inspection_service
    viewmodel._apply_detector_settings()

    assert viewmodel.inspection_service is first_service
    assert len(detectors) == 1

    current = DetectorSettings(detector_backend="onnx")
    viewmodel._apply_detector_settings()

    assert len(detectors) == 2
    assert detectors[0].closed is True
    viewmodel.shutdown()
    assert detectors[1].closed is True
