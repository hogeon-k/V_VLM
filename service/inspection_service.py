from __future__ import annotations

from pathlib import Path
import logging

from config.settings import ERROR_LOG_PATH
from model.inspection_result import InspectionResult
from repository.inspection_repository import InspectionRepository
from service.image_service import ImageService
from service.result_service import ResultService
from service.vlm_service import VlmService
from service.yolo_service import YoloService

logger = logging.getLogger(__name__)


class InspectionService:
    def __init__(
        self,
        image_service: ImageService | None = None,
        yolo_service: YoloService | None = None,
        vlm_service: VlmService | None = None,
        result_service: ResultService | None = None,
        inspection_repository: InspectionRepository | None = None,
    ) -> None:
        self.image_service = image_service or ImageService()
        self.yolo_service = yolo_service or YoloService()
        self.vlm_service = vlm_service or VlmService()
        self.result_service = result_service or ResultService()
        self.inspection_repository = inspection_repository or InspectionRepository()
        _configure_error_logging()

    def inspect(self, image_path: str | Path) -> InspectionResult:
        """Run YOLO first, then call VLM only for NG images."""
        source_path = self.image_service.prepare_image(Path(image_path))

        try:
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
                    logger.exception("VLM failed for %s", source_path)
                    vlm_explanation = f"[VLM error] {exc}"
        except Exception:
            logger.exception("Inspection failed for %s", source_path)
            raise

        inspection_result = InspectionResult(
            source_image_path=source_path,
            result_image_path=yolo_result.annotated_image_path,
            status=status,
            detections=yolo_result.detections,
            vlm_explanation=vlm_explanation,
        )
        self.inspection_repository.save(inspection_result)
        return inspection_result

    def inspect_image(self, image_path: str | Path) -> InspectionResult:
        """Compatibility wrapper for the existing UI/service naming."""
        return self.inspect(image_path)


def _configure_error_logging() -> None:
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == ERROR_LOG_PATH
        for handler in logger.handlers
    ):
        return
    handler = logging.FileHandler(ERROR_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
