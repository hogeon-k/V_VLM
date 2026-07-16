from __future__ import annotations

from pathlib import Path

from model.inspection_result import InspectionResult
from service.inspection_service import InspectionService

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class AutoInspectionService:
    def __init__(self, inspection_service: InspectionService | None = None) -> None:
        self.inspection_service = inspection_service or InspectionService()

    def list_images(self, input_dir: Path) -> list[Path]:
        if not input_dir:
            raise ValueError("이미지 폴더를 선택하세요.")
        directory = Path(input_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {directory}")
        return sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def inspect_directory(self, input_dir: Path) -> list[InspectionResult]:
        images = self.list_images(input_dir)
        if not images:
            raise ValueError("선택한 폴더에 이미지 파일이 없습니다.")
        return [self.inspection_service.inspect_image(image_path) for image_path in images]
