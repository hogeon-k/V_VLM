from __future__ import annotations

from pathlib import Path

from model.inspection_result import InspectionResult
from service.image_service import ImageService
from service.result_service import ResultService
from service.vlm_service import VlmService
from service.yolo_service import YoloService


class InspectionService:
    def __init__(
        self,
        image_service: ImageService | None = None,
        yolo_service: YoloService | None = None,
        vlm_service: VlmService | None = None,
        result_service: ResultService | None = None,
    ) -> None:
        self.image_service = image_service or ImageService()
        self.yolo_service = yolo_service or YoloService()
        self.vlm_service = vlm_service or VlmService()
        self.result_service = result_service or ResultService()

    def inspect_image(self, image_path: Path) -> InspectionResult:
        # TODO: Coordinate image loading, YOLO detection, optional VLM explanation,
        # result image generation, and repository persistence.
        return InspectionResult(image_name=image_path.name, original_image_path=image_path)
