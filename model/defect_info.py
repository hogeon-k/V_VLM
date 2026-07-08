from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DefectInfo:
    defect_type: str
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    vlm_description: str | None = None
