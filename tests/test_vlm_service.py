from pathlib import Path

from model.defect_info import Detection
from model.yolo_result import YoloResult
from service.vlm_service import VlmService


def test_vlm_service_skips_ok_result() -> None:
    description = VlmService().describe_defects(Path("sample.png"), YoloResult(Path("sample.png")))

    assert description is None


def test_vlm_service_generates_description_for_ng_result() -> None:
    class FakeClient:
        def generate(self, prompt: str, image_path: Path) -> str:
            assert "open_circuit" in prompt
            assert image_path == Path("sample.png")
            return "  explanation  "

    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])

    description = VlmService(client=FakeClient()).describe_defects(Path("sample.png"), yolo_result)

    assert description == "explanation"
