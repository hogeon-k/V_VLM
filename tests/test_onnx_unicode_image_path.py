from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from config.detector_settings import DetectorSettings
from image_processing.image_loader import ImageLoader, read_image_unicode
from model.yolo_result import YoloResult
from service.detector_backend_factory import create_yolo_service_from_settings
from service.inspection_service import InspectionService
from service.onnx_detector import OnnxDetector


class PassthroughImageService:
    def prepare_image(self, image_path: Path) -> Path:
        return image_path


class FakeInspectionRepository:
    def save(self, inspection_result: object) -> int:
        return 1


class FakeOnnxSession:
    def run(self, output_names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert output_names == ["output0"]
        input_tensor = inputs["images"]
        assert input_tensor.shape == (1, 3, 960, 960)
        return [np.zeros((1, 7, 1), dtype=np.float32)]

    def get_providers(self) -> list[str]:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


class FakeOnnxDetector(OnnxDetector):
    def _load_session(self) -> FakeOnnxSession:
        self._input_name = "images"
        self._output_name = "output0"
        self._input_shape = [1, 3, 960, 960]
        self.actual_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.using_cuda = True
        return FakeOnnxSession()


def _write_test_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    encoded.tofile(str(path))


def test_unicode_image_path_read_succeeds(tmp_path: Path) -> None:
    image_path = tmp_path / "데이터" / "검사이미지.jpg"
    _write_test_image(image_path)

    image = read_image_unicode(image_path)

    assert image.dtype == np.uint8
    assert image.shape == (12, 16, 3)


def test_unicode_image_path_with_spaces_read_succeeds(tmp_path: Path) -> None:
    image_path = tmp_path / "데이터 폴더" / "검사 이미지.jpg"
    _write_test_image(image_path)

    image = ImageLoader().load(image_path)

    assert image.dtype == np.uint8
    assert image.shape[2] == 3


def test_ascii_image_path_still_reads(tmp_path: Path) -> None:
    image_path = tmp_path / "ascii" / "sample.jpg"
    _write_test_image(image_path)

    assert read_image_unicode(image_path).shape == (12, 16, 3)


def test_missing_image_path_reports_original_path(tmp_path: Path) -> None:
    image_path = tmp_path / "데이터" / "missing.jpg"

    with pytest.raises(FileNotFoundError, match="missing.jpg") as exc_info:
        read_image_unicode(image_path)

    assert str(image_path) in str(exc_info.value)


def test_empty_image_file_reports_original_path(tmp_path: Path) -> None:
    image_path = tmp_path / "데이터" / "empty.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty") as exc_info:
        read_image_unicode(image_path)

    assert str(image_path) in str(exc_info.value)


def test_non_image_file_reports_decode_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "데이터" / "not_image.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="not decodable") as exc_info:
        read_image_unicode(image_path)

    assert str(image_path) in str(exc_info.value)


def test_onnx_detector_uses_unicode_loader_not_cv2_imread() -> None:
    source = Path("service/onnx_detector.py").read_text(encoding="utf-8")

    assert "read_image_unicode(source_path)" in source
    assert "cv2.imread" not in source
    assert "np.fromfile" in Path("image_processing/image_loader.py").read_text(encoding="utf-8")
    assert "cv2.imdecode" in Path("image_processing/image_loader.py").read_text(encoding="utf-8")


def test_onnx_detector_zero_detections_is_valid_ok_result(tmp_path: Path) -> None:
    image_path = tmp_path / "데이터" / "검사 이미지.jpg"
    _write_test_image(image_path)
    detector = FakeOnnxDetector(tmp_path / "best.onnx")

    result = detector.detect(image_path)

    assert isinstance(result, YoloResult)
    assert result.image_path == image_path
    assert result.detections == []
    assert result.is_ng is False


def test_inspection_service_onnx_zero_detections_returns_ok(tmp_path: Path) -> None:
    image_path = tmp_path / "데이터" / "검사 이미지.jpg"
    _write_test_image(image_path)
    detector = FakeOnnxDetector(tmp_path / "best.onnx")

    result = InspectionService(
        image_service=PassthroughImageService(),
        yolo_service=type("FakeYoloService", (), {"detect": detector.detect})(),
        inspection_repository=FakeInspectionRepository(),
    ).inspect_image(image_path)

    assert result.status == "OK"
    assert result.defect_count == 0
    assert result.image_name == "검사 이미지.jpg"


def test_backend_factory_onnx_creation_still_returns_python_onnx_detector() -> None:
    service = create_yolo_service_from_settings(DetectorSettings(detector_backend="onnx"))

    assert isinstance(service.detector, OnnxDetector)
    assert service.detector.imgsz == 960
    assert service.detector.conf == 0.15
    assert service.detector.iou == 0.5
    assert service.detector.requested_provider == "CUDAExecutionProvider"


def test_tensorrt_adapter_subprocess_decode_path_unchanged() -> None:
    source = Path("service/tensorrt_detector_adapter.py").read_text(encoding="utf-8")

    assert "text=False" in source
    assert "_decode_process_output" in source
