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

    def inspect(self, image_path: str | Path) -> InspectionResult:
        """Run YOLO first, then call VLM only for NG images."""
        source_path = Path(image_path)
        yolo_result = self.yolo_service.detect(source_path)

        status = "NG" if yolo_result.is_ng else "OK"
        vlm_explanation: str | None = None
        if yolo_result.is_ng:
            try:
                vlm_explanation = self.vlm_service.describe_defects(
                    yolo_result.annotated_image_path or source_path,
                    yolo_result,
                )
            except RuntimeError as exc:
                vlm_explanation = f"[VLM error] {exc}"

        return InspectionResult(
            source_image_path=source_path,
            result_image_path=yolo_result.annotated_image_path,
            status=status,
            detections=yolo_result.detections,
            vlm_explanation=vlm_explanation,
        )

    def inspect_image(self, image_path: str | Path) -> InspectionResult:
        """Compatibility wrapper for the existing UI/service naming."""
        return self.inspect(image_path)
