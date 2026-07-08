from pathlib import Path

from model.yolo_result import YoloResult
from service.vlm_service import VlmService


def test_vlm_service_has_provider_neutral_placeholder() -> None:
    description = VlmService().describe_defects(Path("sample.png"), YoloResult(Path("sample.png")))

    assert description is None
