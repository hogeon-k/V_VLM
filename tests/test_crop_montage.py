from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from model.defect_info import Detection
from vlm.crop_montage import (
    create_crop_montage_jpeg_bytes,
    create_crop_montage_result,
    create_detection_crops,
    save_montage_bytes,
)


def detection(index: int, x1: int, y1: int, x2: int, y2: int) -> Detection:
    return Detection(
        class_id=index,
        class_name=f"defect_{index}",
        confidence=0.8 + index / 100,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        location="중단 오른쪽",
    )


def test_detection_crops_use_original_image_and_minimum_size(tmp_path) -> None:
    image_path = tmp_path / "board.jpg"
    Image.new("RGB", (1000, 800), "white").save(image_path)

    crops = create_detection_crops(
        image_path,
        [detection(1, 490, 390, 500, 400)],
        padding=32,
        min_crop_size=256,
        max_crop_size=512,
    )

    crop = crops[0]
    assert crop.width == 256
    assert crop.height == 256
    assert crop.crop_box == (367, 267, 623, 523)
    assert crop.bbox_in_crop == (123, 123, 133, 133)
    assert crop.label.startswith("D1 | defect_1")


def test_detection_crop_is_clamped_to_image_boundary(tmp_path) -> None:
    image_path = tmp_path / "board.jpg"
    Image.new("RGB", (300, 300), "white").save(image_path)

    crops = create_detection_crops(
        image_path,
        [detection(1, 0, 0, 20, 20)],
        padding=32,
        min_crop_size=256,
        max_crop_size=512,
    )

    assert crops[0].crop_box == (0, 0, 256, 256)
    assert crops[0].bbox_in_crop == (0, 0, 20, 20)


def test_detection_crop_respects_max_size_and_order(tmp_path) -> None:
    image_path = tmp_path / "board.jpg"
    Image.new("RGB", (1200, 1000), "white").save(image_path)

    crops = create_detection_crops(
        image_path,
        [
            detection(1, 100, 100, 200, 180),
            detection(2, 600, 400, 660, 460),
        ],
        max_crop_size=512,
    )

    assert [crop.detection_index for crop in crops] == [1, 2]
    assert all(crop.width <= 512 for crop in crops)
    assert all(crop.height <= 512 for crop in crops)
    assert crops[0].label.startswith("D1 | defect_1")
    assert crops[1].label.startswith("D2 | defect_2")


@pytest.mark.parametrize("count", [1, 2, 3])
def test_crop_montage_jpeg_bytes_for_detection_counts(tmp_path, count: int) -> None:
    image_path = tmp_path / "board.jpg"
    Image.new("RGB", (1000, 800), "white").save(image_path)
    detections = [
        detection(index, 100 * index, 80 * index, 100 * index + 30, 80 * index + 30)
        for index in range(1, count + 1)
    ]

    montage_bytes = create_crop_montage_jpeg_bytes(
        image_path,
        detections,
        max_size=960,
        columns=2,
    )

    assert montage_bytes
    with Image.open(BytesIO(montage_bytes)) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert max(image.size) <= 960


def test_crop_montage_result_includes_metadata(tmp_path) -> None:
    image_path = tmp_path / "board.jpg"
    Image.new("RGB", (1000, 800), "white").save(image_path)

    result = create_crop_montage_result(
        image_path,
        [detection(1, 100, 80, 130, 110)],
        max_size=960,
    )

    assert result.image_bytes
    assert result.width > 0
    assert result.height > 0
    assert result.crop_count == 1


def test_crop_montage_result_respects_size_limit_without_losing_crop_count(tmp_path) -> None:
    image_path = tmp_path / "board.jpg"
    Image.new("RGB", (1800, 1200), "white").save(image_path)
    detections = [
        detection(1, 100, 80, 150, 130),
        detection(2, 700, 400, 760, 470),
        detection(3, 1300, 900, 1360, 980),
        detection(4, 1500, 100, 1580, 180),
    ]

    result = create_crop_montage_result(
        image_path,
        detections,
        max_size=640,
        columns=2,
    )

    assert result.crop_count == 4
    assert max(result.width, result.height) == 640
    assert result.width == result.height
    with Image.open(BytesIO(result.image_bytes)) as image:
        assert image.size == (640, 640)


def test_crop_montage_result_does_not_upscale_small_montage(tmp_path) -> None:
    image_path = tmp_path / "board.jpg"
    Image.new("RGB", (180, 180), "white").save(image_path)

    result = create_crop_montage_result(
        image_path,
        [detection(1, 20, 20, 40, 40)],
        max_size=640,
        min_crop_size=64,
        max_crop_size=128,
    )

    assert result.crop_count == 1
    assert max(result.width, result.height) < 640


def test_crop_montage_rejects_empty_detections(tmp_path) -> None:
    image_path = tmp_path / "board.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)

    with pytest.raises(ValueError, match="At least one detection"):
        create_crop_montage_jpeg_bytes(image_path, [])


def test_detection_crops_reject_missing_image(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="VLM crop source image not found"):
        create_detection_crops(tmp_path / "missing.jpg", [detection(1, 1, 2, 3, 4)])


def test_save_montage_bytes_creates_parent_and_valid_jpeg(tmp_path) -> None:
    image_path = tmp_path / "board.jpg"
    Image.new("RGB", (1000, 800), "white").save(image_path)
    montage_bytes = create_crop_montage_jpeg_bytes(
        image_path,
        [detection(1, 100, 80, 130, 110)],
    )

    saved_path = save_montage_bytes(montage_bytes, tmp_path / "nested" / "montage")

    assert saved_path == tmp_path / "nested" / "montage.jpg"
    assert saved_path.stat().st_size > 0
    with Image.open(saved_path) as image:
        assert image.format == "JPEG"


def test_save_montage_bytes_rejects_empty_bytes(tmp_path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        save_montage_bytes(b"", tmp_path / "montage.jpg")
