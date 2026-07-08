from __future__ import annotations

from pathlib import Path

from config.settings import RESULT_IMAGE_DIR
from image_processing.bbox_drawer import BBoxDrawer
from model.inspection_result import InspectionResult


class ResultService:
    def __init__(self, bbox_drawer: BBoxDrawer | None = None) -> None:
        self.bbox_drawer = bbox_drawer or BBoxDrawer()

    def build_result_path(self, image_path: Path) -> Path:
        # TODO: Add collision handling when saving generated result images.
        return RESULT_IMAGE_DIR / image_path.name

    def save_result(self, inspection_result: InspectionResult) -> InspectionResult:
        # TODO: Draw bounding boxes and save the visual result under data/result_images/.
        return inspection_result
