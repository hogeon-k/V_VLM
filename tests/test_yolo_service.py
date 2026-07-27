from pathlib import Path

import numpy as np
import pytest

from model.yolo_result import YoloResult
from service.yolo_service import YoloService
from yolo.detector import YoloDetector
from yolo.model_loader import YoloModelLoader
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


def test_yolo_model_loader_rejects_missing_model(tmp_path: Path) -> None:
    loader = YoloModelLoader(YoloConfig(model_path=tmp_path / "missing.pt"))

    with pytest.raises(FileNotFoundError, match="YOLO model file not found"):
        loader.load()


def test_yolo_detector_rejects_directory_before_model_load(tmp_path: Path) -> None:
    class FailingLoader:
        def load(self) -> object:
            raise AssertionError("Model must not load for a directory input.")

    detector = YoloDetector(model_loader=FailingLoader())

    with pytest.raises(FileNotFoundError, match="Input image not found"):
        detector.detect(tmp_path)


def test_yolo_detector_passes_config_and_saves_unicode_output(tmp_path: Path) -> None:
    class Values:
        def __init__(self, values: list[object]) -> None:
            self.values = values

        def cpu(self) -> "Values":
            return self

        def tolist(self) -> list[object]:
            return self.values

    class Boxes:
        xyxy = Values([[1.2, 2.4, 10.6, 12.8]])
        conf = Values([0.91])
        cls = Values([0.0])

        def __len__(self) -> int:
            return 1

    class Result:
        boxes = Boxes()
        names = {0: "open_circuit"}
        orig_shape = (20, 30)

        def plot(self) -> np.ndarray:
            return np.zeros((20, 30, 3), dtype=np.uint8)

    class Model:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def predict(self, **kwargs: object) -> list[Result]:
            self.kwargs = kwargs
            return [Result()]

    class Loader:
        def __init__(self, model: Model) -> None:
            self.model = model

        def load(self) -> Model:
            return self.model

    source = tmp_path / "입력.png"
    source.write_bytes(b"fake")
    output = tmp_path / "결과.png"
    model = Model()
    config = YoloConfig(
        model_path=tmp_path / "unused.pt",
        confidence_threshold=0.25,
        image_size=640,
        iou_threshold=0.6,
        device="cpu",
    )

    result = YoloDetector(model_loader=Loader(model), config=config).detect(source, output)

    assert isinstance(result, YoloResult)
    assert result.annotated_image_path == output
    assert output.is_file()
    assert model.kwargs == {
        "source": str(source),
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.6,
        "device": "cpu",
        "save": False,
        "verbose": False,
    }
    assert result.detections[0].class_name == "open_circuit"
    assert result.detections[0].confidence == pytest.approx(0.91)
    assert (result.detections[0].x1, result.detections[0].y1) == (1, 2)
    assert (result.detections[0].x2, result.detections[0].y2) == (11, 13)


def test_yolo_detector_converts_empty_boxes_to_empty_detections() -> None:
    class EmptyBoxes:
        def __len__(self) -> int:
            return 0

    class Result:
        boxes = EmptyBoxes()

    assert YoloDetector()._to_detections(Result()) == []
