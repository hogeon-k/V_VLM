from pathlib import Path

from model.defect_info import Detection
from model.yolo_result import YoloResult
from service.inspection_service import InspectionService


def test_inspection_service_returns_ok_without_vlm() -> None:
    class FakeYoloService:
        def detect(self, image_path: Path) -> YoloResult:
            return YoloResult(image_path=image_path, annotated_image_path=Path("result.jpg"))

    class FailingVlmService:
        def describe_defects(self, image_path: Path, yolo_result: YoloResult) -> str:
            raise AssertionError("VLM should not be called for OK images.")

    result = InspectionService(
        yolo_service=FakeYoloService(),
        vlm_service=FailingVlmService(),
    ).inspect_image(Path("sample.png"))

    assert result.image_name == "sample.png"
    assert result.defect_count == 0
    assert result.status == "OK"
    assert result.vlm_explanation is None


def test_inspection_service_returns_ng_with_vlm_description() -> None:
    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)

    class FakeYoloService:
        def detect(self, image_path: Path) -> YoloResult:
            return YoloResult(
                image_path=image_path,
                detections=[detection],
                annotated_image_path=Path("result.jpg"),
            )

    class FakeVlmService:
        def describe_defects(self, image_path: Path, yolo_result: YoloResult) -> str:
            assert image_path == Path("result.jpg")
            return "VLM explanation"

    result = InspectionService(
        yolo_service=FakeYoloService(),
        vlm_service=FakeVlmService(),
    ).inspect(Path("sample.png"))

    assert result.status == "NG"
    assert result.defect_count == 1
    assert result.vlm_explanation == "VLM explanation"
