from __future__ import annotations

from pathlib import Path

from model.yolo_result import YoloResult
from yolo.detector import YoloDetector


class YoloService:
    def __init__(self, detector: YoloDetector | None = None) -> None:
        self.detector = detector or YoloDetector()

    def detect(self, image_path: str | Path, output_path: str | Path | None = None) -> YoloResult:
        """Delegate image detection to the configured YOLO detector."""
        return self.detector.detect(image_path, output_path=output_path)
