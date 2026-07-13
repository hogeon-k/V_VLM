from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from vlm.image_preprocessor import resize_image_to_jpeg_bytes


def read_jpeg_size(image_bytes: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(image_bytes)) as image:
        return image.size


def test_resize_large_image_to_max_side(tmp_path) -> None:
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (1920, 480), "white").save(image_path)

    image_bytes = resize_image_to_jpeg_bytes(image_path, max_size=960)

    assert read_jpeg_size(image_bytes) == (960, 240)


def test_resize_keeps_aspect_ratio_for_tall_image(tmp_path) -> None:
    image_path = tmp_path / "tall.png"
    Image.new("RGB", (300, 1200), "white").save(image_path)

    image_bytes = resize_image_to_jpeg_bytes(image_path, max_size=960)

    assert read_jpeg_size(image_bytes) == (240, 960)


def test_small_image_is_not_upscaled(tmp_path) -> None:
    image_path = tmp_path / "small.png"
    Image.new("RGB", (320, 200), "white").save(image_path)

    image_bytes = resize_image_to_jpeg_bytes(image_path, max_size=960)

    assert read_jpeg_size(image_bytes) == (320, 200)


def test_rgba_image_is_converted_to_rgb_jpeg(tmp_path) -> None:
    image_path = tmp_path / "rgba.png"
    Image.new("RGBA", (100, 80), (255, 0, 0, 128)).save(image_path)

    image_bytes = resize_image_to_jpeg_bytes(image_path, max_size=960)

    with Image.open(BytesIO(image_bytes)) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (100, 80)
    assert image_bytes


def test_missing_image_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="VLM source image not found"):
        resize_image_to_jpeg_bytes(tmp_path / "missing.png")
