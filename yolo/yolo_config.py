from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.settings import YOLO_MODEL_PATH


@dataclass(frozen=True, slots=True)
class YoloConfig:
    model_path: Path = YOLO_MODEL_PATH
    confidence_threshold: float = 0.25
    image_size: int = 640
