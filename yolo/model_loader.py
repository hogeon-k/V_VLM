from __future__ import annotations

from pathlib import Path

from yolo.yolo_config import YoloConfig


class YoloModelLoader:
    def __init__(self, config: YoloConfig | None = None) -> None:
        self.config = config or YoloConfig()
        self._model: object | None = None

    def load(self) -> object:
        """Load and cache the Ultralytics YOLO model."""
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")

        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "ultralytics is not installed. Run: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
                ) from exc

            self._model = YOLO(str(model_path))

        return self._model
