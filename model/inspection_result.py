from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from model.defect_info import DefectInfo, Detection


@dataclass(slots=True)
class YoloDetectionResult:
    """Structured YOLO result requested by the terminal inspection flow."""

    is_ng: bool
    detections: list[Detection]
    annotated_image_path: Path


@dataclass(slots=True)
class InspectionResult:
    source_image_path: Path
    id: int | None = None
    result_image_path: Path | None = None
    status: str = "PENDING"
    detections: list[Detection] = field(default_factory=list)
    vlm_explanation: str | None = None
    inspected_at: datetime | None = None

    @property
    def image_name(self) -> str:
        return self.source_image_path.name

    @property
    def original_image_path(self) -> Path:
        return self.source_image_path

    @property
    def defects(self) -> list[Detection]:
        return self.detections

    @property
    def vlm_description(self) -> str | None:
        return self.vlm_explanation

    @vlm_description.setter
    def vlm_description(self, value: str | None) -> None:
        self.vlm_explanation = value

    @property
    def defect_count(self) -> int:
        return len(self.detections)
