from __future__ import annotations

from pathlib import Path

from service.inspection_service import InspectionService


class AutoInspectionService:
    def __init__(self, inspection_service: InspectionService | None = None) -> None:
        self.inspection_service = inspection_service or InspectionService()

    def inspect_directory(self, input_dir: Path) -> list[object]:
        # TODO: Scan input images and run inspections without blocking the UI thread.
        raise NotImplementedError
