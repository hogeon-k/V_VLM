from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image


def resize_image_to_jpeg_bytes(
    image_path: str | Path,
    max_size: int = 960,
    quality: int = 90,
) -> bytes:
    """Resize an image in memory and return RGB JPEG bytes for VLM input."""
    resolved_path = Path(image_path).resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(f"VLM source image not found: {resolved_path}")

    with Image.open(resolved_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size))

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)

    image_bytes = buffer.getvalue()
    if not image_bytes:
        raise RuntimeError("Failed to create resized VLM image bytes.")

    return image_bytes


def read_image_size_from_bytes(image_bytes: bytes) -> tuple[int, int]:
    """Read image dimensions from in-memory image bytes."""
    if not image_bytes:
        raise ValueError("image_bytes is empty.")

    with Image.open(BytesIO(image_bytes)) as image:
        return image.size
