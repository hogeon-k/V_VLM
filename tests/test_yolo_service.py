from pathlib import Path

from service.yolo_service import YoloService


def test_yolo_service_returns_empty_scaffold_result() -> None:
    result = YoloService().detect(Path("sample.png"))

    assert result.image_path == Path("sample.png")
    assert result.defect_count == 0
