from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from model.defect_info import Detection


@dataclass(frozen=True)
class PreparedDetectionCrop:
    """One in-memory crop prepared for VLM defect inspection."""

    detection_index: int
    image: Image.Image
    label: str
    width: int
    height: int
    crop_box: tuple[int, int, int, int]
    bbox_in_crop: tuple[int, int, int, int]


@dataclass(frozen=True)
class CropMontageResult:
    """JPEG crop montage bytes plus metadata used by the VLM pipeline."""

    image_bytes: bytes
    width: int
    height: int
    crop_count: int


def create_detection_crops(
    image_path: str | Path,
    detections: Sequence[Detection],
    padding: int = 192,
    min_crop_size: int = 256,
    max_crop_size: int = 512,
) -> list[PreparedDetectionCrop]:
    """Create labeled in-memory crops around YOLO detections."""
    if not detections:
        raise ValueError("At least one detection is required to create VLM crops.")

    resolved_path = Path(image_path).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"VLM crop source image not found: {resolved_path}")

    with Image.open(resolved_path) as source:
        source_image = source.convert("RGB")
        crops = [
            _create_one_crop(
                source_image,
                detection=detection,
                detection_index=index,
                padding=padding,
                min_crop_size=min_crop_size,
                max_crop_size=max_crop_size,
            )
            for index, detection in enumerate(detections, start=1)
        ]

    return crops


def create_crop_montage_result(
    image_path: str | Path,
    detections: Sequence[Detection],
    max_size: int = 960,
    quality: int = 90,
    columns: int = 2,
    padding: int = 192,
    min_crop_size: int = 256,
    max_crop_size: int = 512,
) -> CropMontageResult:
    """Combine detection crops into one RGB JPEG montage for VLM input."""
    crops = create_detection_crops(
        image_path=image_path,
        detections=detections,
        padding=padding,
        min_crop_size=min_crop_size,
        max_crop_size=max_crop_size,
    )
    if not crops:
        raise RuntimeError("No VLM crops were created.")

    columns = max(1, columns)
    tile_width = max(crop.width for crop in crops)
    tile_height = max(crop.height for crop in crops)
    rows = ceil(len(crops) / columns)

    montage = Image.new(
        "RGB",
        (tile_width * columns, tile_height * rows),
        color=(246, 246, 246),
    )

    for index, crop in enumerate(crops):
        row = index // columns
        column = index % columns
        x = column * tile_width + (tile_width - crop.width) // 2
        y = row * tile_height + (tile_height - crop.height) // 2
        montage.paste(crop.image.convert("RGB"), (x, y))

    montage.thumbnail((max_size, max_size))

    buffer = BytesIO()
    montage.save(buffer, format="JPEG", quality=quality)
    montage_bytes = buffer.getvalue()
    if not montage_bytes:
        raise RuntimeError("Failed to create VLM crop montage JPEG bytes.")

    return CropMontageResult(
        image_bytes=montage_bytes,
        width=montage.width,
        height=montage.height,
        crop_count=len(crops),
    )


def create_crop_montage_jpeg_bytes(
    image_path: str | Path,
    detections: Sequence[Detection],
    max_size: int = 960,
    quality: int = 90,
    columns: int = 2,
    padding: int = 192,
    min_crop_size: int = 256,
    max_crop_size: int = 512,
) -> bytes:
    """Return only the JPEG bytes for callers that use the original API."""
    return create_crop_montage_result(
        image_path=image_path,
        detections=detections,
        max_size=max_size,
        quality=quality,
        columns=columns,
        padding=padding,
        min_crop_size=min_crop_size,
        max_crop_size=max_crop_size,
    ).image_bytes


def save_montage_bytes(image_bytes: bytes, output_path: Path) -> Path:
    """Save already-created JPEG montage bytes and return the final path."""
    if not image_bytes:
        raise ValueError("Crop montage image bytes must not be empty.")

    target_path = Path(output_path)
    if target_path.suffix.lower() not in {".jpg", ".jpeg"}:
        target_path = target_path.with_suffix(".jpg")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(image_bytes)
    return target_path


def _create_one_crop(
    source_image: Image.Image,
    detection: Detection,
    detection_index: int,
    padding: int,
    min_crop_size: int,
    max_crop_size: int,
) -> PreparedDetectionCrop:
    image_width, image_height = source_image.size
    x1, y1, x2, y2 = _normalized_box(detection)
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    crop_width = min(max(box_width * 8, padding * 2, min_crop_size), max_crop_size)
    crop_height = min(max(box_height * 8, padding * 2, min_crop_size), max_crop_size)
    crop_box = _centered_crop_box(
        center_x=center_x,
        center_y=center_y,
        crop_width=int(round(crop_width)),
        crop_height=int(round(crop_height)),
        image_width=image_width,
        image_height=image_height,
    )
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop_image = source_image.crop(crop_box)
    bbox_in_crop = (
        _clamp(x1 - crop_left, 0, crop_image.width),
        _clamp(y1 - crop_top, 0, crop_image.height),
        _clamp(x2 - crop_left, 0, crop_image.width),
        _clamp(y2 - crop_top, 0, crop_image.height),
    )
    label = _build_label(detection_index, detection)
    _draw_crop_overlay(crop_image, bbox_in_crop, label)

    return PreparedDetectionCrop(
        detection_index=detection_index,
        image=crop_image,
        label=label,
        width=crop_image.width,
        height=crop_image.height,
        crop_box=crop_box,
        bbox_in_crop=bbox_in_crop,
    )


def _normalized_box(detection: Detection) -> tuple[int, int, int, int]:
    x1 = int(round(detection.x1))
    y1 = int(round(detection.y1))
    x2 = int(round(detection.x2))
    y2 = int(round(detection.y2))
    if x2 < x1 or y2 < y1:
        raise ValueError("Detection bounding box must satisfy x1 <= x2 and y1 <= y2.")
    return x1, y1, x2, y2


def _centered_crop_box(
    center_x: float,
    center_y: float,
    crop_width: int,
    crop_height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    crop_width = min(crop_width, image_width)
    crop_height = min(crop_height, image_height)
    left = int(round(center_x - crop_width / 2))
    top = int(round(center_y - crop_height / 2))
    left = _clamp(left, 0, image_width - crop_width)
    top = _clamp(top, 0, image_height - crop_height)
    return left, top, left + crop_width, top + crop_height


def _build_label(detection_index: int, detection: Detection) -> str:
    parts = [
        f"D{detection_index}",
        detection.class_name,
        f"{detection.confidence:.3f}",
    ]
    if detection.location:
        parts.insert(2, detection.location)
    return " | ".join(parts)


def _draw_crop_overlay(
    image: Image.Image,
    bbox_in_crop: tuple[int, int, int, int],
    label: str,
) -> None:
    draw = ImageDraw.Draw(image)
    line_width = max(2, min(image.size) // 128)
    draw.rectangle(bbox_in_crop, outline=(255, 32, 32), width=line_width)

    font = ImageFont.load_default()
    draw_label = _font_safe_label(draw, label, font)
    text_bbox = draw.textbbox((0, 0), draw_label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    pad = 4
    label_left = 4
    label_top = 4
    draw.rectangle(
        (
            label_left,
            label_top,
            min(image.width - 1, label_left + text_width + pad * 2),
            min(image.height - 1, label_top + text_height + pad * 2),
        ),
        fill=(255, 255, 255),
        outline=(255, 32, 32),
    )
    draw.text(
        (label_left + pad, label_top + pad),
        draw_label,
        fill=(20, 20, 20),
        font=font,
    )


def _font_safe_label(draw: ImageDraw.ImageDraw, label: str, font: ImageFont.ImageFont) -> str:
    try:
        draw.textbbox((0, 0), label, font=font)
    except UnicodeEncodeError:
        return label.encode("ascii", errors="replace").decode("ascii")
    return label


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))
