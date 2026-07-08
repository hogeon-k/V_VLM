from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from model.defect_info import DefectInfo


@dataclass(slots=True)
class InspectionResult:
    image_name: str
    original_image_path: Path
    result_image_path: Path | None = None
    status: str = "PENDING"
    defects: list[DefectInfo] = field(default_factory=list)
    vlm_description: str | None = None
    inspected_at: datetime | None = None

    @property
    def defect_count(self) -> int:
        return len(self.defects)
