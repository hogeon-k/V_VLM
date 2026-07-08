from __future__ import annotations

from pathlib import Path

from model.inspection_result import InspectionResult
from service.inspection_service import InspectionService


class InspectionViewModel:
    def __init__(self, inspection_service: InspectionService | None = None) -> None:
        self.inspection_service = inspection_service or InspectionService()
        self.current_result: InspectionResult | None = None

    def inspect_image(self, image_path: Path) -> InspectionResult:
        # TODO: Run this through a worker thread before connecting to the real UI.
        self.current_result = self.inspection_service.inspect_image(image_path)
        return self.current_result
