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
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")
        return path
