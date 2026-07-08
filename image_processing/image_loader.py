from __future__ import annotations

from pathlib import Path


class ImageLoader:
    def load(self, image_path: Path) -> object:
        # TODO: Load images with OpenCV or Pillow and validate readable input.
        raise NotImplementedError
