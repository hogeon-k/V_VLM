from __future__ import annotations

from pathlib import Path

from image_processing.image_loader import ImageLoader
from image_processing.preprocessor import ImagePreprocessor


class ImageService:
    def __init__(
        self,
        image_loader: ImageLoader | None = None,
        preprocessor: ImagePreprocessor | None = None,
    ) -> None:
        self.image_loader = image_loader or ImageLoader()
        self.preprocessor = preprocessor or ImagePreprocessor()

    def prepare_image(self, image_path: Path) -> Path:
        # TODO: Validate and preprocess images before inference.
        return image_path
