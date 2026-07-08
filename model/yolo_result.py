from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from model.defect_info import DefectInfo


@dataclass(slots=True)
class YoloResult:
    image_path: Path
    defects: list[DefectInfo] = field(default_factory=list)

    @property
    def defect_count(self) -> int:
        return len(self.defects)
