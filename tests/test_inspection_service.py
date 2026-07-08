from pathlib import Path

from service.inspection_service import InspectionService


def test_inspection_service_returns_pending_result() -> None:
    result = InspectionService().inspect_image(Path("sample.png"))

    assert result.image_name == "sample.png"
    assert result.defect_count == 0
    assert result.status == "PENDING"
