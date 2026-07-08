from __future__ import annotations

from pathlib import Path

from model.yolo_result import YoloResult
from yolo.detector import YoloDetector


class YoloService:
    def __init__(self, detector: YoloDetector | None = None) -> None:
        self.detector = detector or YoloDetector()

    def detect(self, image_path: Path) -> YoloResult:
        # TODO: Delegate to YOLO detector after model configuration is finalized.
        return YoloResult(image_path=image_path)
