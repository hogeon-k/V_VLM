from __future__ import annotations

from pathlib import Path

from model.yolo_result import YoloResult
from yolo.model_loader import YoloModelLoader
from yolo.yolo_config import YoloConfig


class YoloDetector:
    def __init__(
        self,
        model_loader: YoloModelLoader | None = None,
        config: YoloConfig | None = None,
    ) -> None:
        self.config = config or YoloConfig()
        self.model_loader = model_loader or YoloModelLoader(self.config)

    def detect(self, image_path: Path) -> YoloResult:
        # TODO: Run YOLO inference and convert detections into DefectInfo records.
        return YoloResult(image_path=image_path)
