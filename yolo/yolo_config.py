from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.settings import YOLO_MODEL_PATH


@dataclass(frozen=True, slots=True)
class YoloConfig:
    model_path: Path = YOLO_MODEL_PATH
    confidence_threshold: float = 0.15
    image_size: int = 960
    iou_threshold: float = 0.7
    device: str = "0"
