from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Detection:
    """One YOLO detection box in pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    location: str | None = None
    vlm_description: str | None = None

    @property
    def defect_type(self) -> str:
        return self.class_name

    @property
    def bbox_x1(self) -> int:
        return self.x1

    @property
    def bbox_y1(self) -> int:
        return self.y1

    @property
    def bbox_x2(self) -> int:
        return self.x2

    @property
    def bbox_y2(self) -> int:
        return self.y2


class DefectInfo(Detection):
    """Backward-compatible defect record used by older services."""

    def __init__(
        self,
        defect_type: str,
        confidence: float,
        bbox_x1: int,
        bbox_y1: int,
        bbox_x2: int,
        bbox_y2: int,
        location: str | None = None,
        vlm_description: str | None = None,
        class_id: int = -1,
    ) -> None:
        super().__init__(
            class_id=class_id,
            class_name=defect_type,
            confidence=confidence,
            x1=bbox_x1,
            y1=bbox_y1,
            x2=bbox_x2,
            y2=bbox_y2,
            location=location,
            vlm_description=vlm_description,
        )
