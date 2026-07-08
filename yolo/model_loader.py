from __future__ import annotations

from yolo.yolo_config import YoloConfig


class YoloModelLoader:
    def __init__(self, config: YoloConfig | None = None) -> None:
        self.config = config or YoloConfig()
        self._model: object | None = None

    def load(self) -> object:
        # TODO: Load and cache the Ultralytics YOLO model from self.config.model_path.
        raise NotImplementedError
