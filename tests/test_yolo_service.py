from pathlib import Path

import pytest

from service.yolo_service import YoloService
from yolo.detector import YoloDetector
from yolo.yolo_config import YoloConfig


def test_yolo_service_delegates_to_detector() -> None:
    class FakeDetector:
        def detect(self, image_path: Path, output_path: Path | None = None) -> object:
            return {"image_path": image_path, "output_path": output_path}

    result = YoloService(detector=FakeDetector()).detect(Path("sample.png"), Path("out.jpg"))

    assert result == {"image_path": Path("sample.png"), "output_path": Path("out.jpg")}


def test_yolo_detector_rejects_missing_image_before_model_load(tmp_path: Path) -> None:
    detector = YoloDetector(config=YoloConfig(model_path=tmp_path / "missing.pt"))

    with pytest.raises(FileNotFoundError, match="Input image not found"):
        detector.detect(tmp_path / "missing.png")
