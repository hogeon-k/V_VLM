from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from model.defect_info import Detection


@dataclass(slots=True)
class YoloResult:
    """YOLO detection output plus the annotated result image path."""

    image_path: Path
    detections: list[Detection] = field(default_factory=list)
    annotated_image_path: Path | None = None

    @property
    def defect_count(self) -> int:
        return len(self.detections)

    @property
    def defects(self) -> list[Detection]:
        return self.detections

    @property
    def is_ng(self) -> bool:
        return bool(self.detections)
