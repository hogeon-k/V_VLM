from __future__ import annotations

from pathlib import Path
import logging

from config.settings import ERROR_LOG_PATH
from model.inspection_result import InspectionResult
from model.inspection_result import (
    VLM_STATUS_COMPLETED,
    VLM_STATUS_FAILED,
    VLM_STATUS_NOT_REQUESTED,
)
from model.yolo_result import YoloResult
from repository.inspection_repository import InspectionRepository
from service.image_service import ImageService
from service.result_service import ResultService
from service.vlm_service import VlmService
from service.yolo_service import YoloService

logger = logging.getLogger(__name__)


class InspectionNotFoundError(ValueError):
    """Raised when an inspection history row cannot be found."""


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
        """Run the legacy full inspection flow: YOLO first, then VLM for NG images."""
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

    def inspect_yolo_only(self, image_path: str | Path) -> InspectionResult:
        """Run YOLO, persist the inspection, and leave VLM generation for history."""
        source_path = self.image_service.prepare_image(Path(image_path))

        try:
            yolo_result = self.yolo_service.detect(source_path)
            status = "NG" if yolo_result.is_ng else "OK"
        except Exception:
            logger.exception("YOLO inspection failed for %s", source_path)
            raise

        inspection_result = InspectionResult(
            source_image_path=source_path,
            result_image_path=yolo_result.annotated_image_path,
            status=status,
            detections=yolo_result.detections,
            vlm_explanation=None,
            vlm_status=VLM_STATUS_NOT_REQUESTED,
        )
        self.inspection_repository.save(inspection_result)
        return inspection_result

    def run_vlm_for_inspection(self, inspection_id: int) -> InspectionResult:
        """Generate a VLM explanation for an already-saved NG inspection."""
        history = self.inspection_repository.find_by_id(inspection_id)
        if history is None:
            raise InspectionNotFoundError(f"Inspection history not found: {inspection_id}")
        if history.status != "NG" or history.defect_count == 0:
            raise ValueError("VLM can only be generated for NG inspections.")

        image_path = history.result_image_path or history.source_image_path
        if not image_path or not Path(image_path).is_file():
            message = f"VLM source image not found: {image_path}"
            self.inspection_repository.update_vlm_result(
                inspection_id,
                VLM_STATUS_FAILED,
                history.vlm_description,
                message,
            )
            raise FileNotFoundError(message)

        if not self.inspection_repository.try_mark_vlm_processing(inspection_id):
            raise RuntimeError("VLM generation is already processing for this inspection.")

        yolo_result = YoloResult(
            image_path=history.source_image_path,
            detections=history.detections,
            annotated_image_path=history.result_image_path,
        )
        try:
            description = self.vlm_service.describe_defects(Path(image_path), yolo_result)
        except Exception as exc:
            logger.exception("VLM failed for inspection_id=%s", inspection_id)
            self.inspection_repository.update_vlm_result(
                inspection_id,
                VLM_STATUS_FAILED,
                history.vlm_description,
                str(exc),
            )
            raise

        if not self.inspection_repository.update_vlm_result(
            inspection_id,
            VLM_STATUS_COMPLETED,
            description,
            None,
        ):
            raise InspectionNotFoundError(f"Inspection history not found: {inspection_id}")
        updated = self.inspection_repository.find_by_id(inspection_id)
        if updated is None:
            raise InspectionNotFoundError(f"Inspection history not found: {inspection_id}")
        return updated

    def inspect_image(self, image_path: str | Path) -> InspectionResult:
        """UI-facing inspection path: return immediately after YOLO is saved."""
        return self.inspect_yolo_only(image_path)


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
