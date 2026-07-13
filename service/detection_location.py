from __future__ import annotations


def calculate_detection_location(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> str:
    """Classify a bounding box center into a 3x3 image region."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be greater than 0.")

    x1, y1, x2, y2 = box
    if x2 < x1 or y2 < y1:
        raise ValueError("Bounding box coordinates must satisfy x1 <= x2 and y1 <= y2.")

    clamped_x1 = _clamp(x1, 0, image_width)
    clamped_y1 = _clamp(y1, 0, image_height)
    clamped_x2 = _clamp(x2, 0, image_width)
    clamped_y2 = _clamp(y2, 0, image_height)

    center_x = (clamped_x1 + clamped_x2) / 2
    center_y = (clamped_y1 + clamped_y2) / 2

    vertical = _vertical_region(center_y, image_height)
    horizontal = _horizontal_region(center_x, image_width)
    return f"{vertical} {horizontal}"


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _horizontal_region(center_x: float, image_width: int) -> str:
    if center_x < image_width / 3:
        return "좌측"
    if center_x < image_width * 2 / 3:
        return "중앙"
    return "오른쪽"


def _vertical_region(center_y: float, image_height: int) -> str:
    if center_y < image_height / 3:
        return "상단"
    if center_y < image_height * 2 / 3:
        return "중단"
    return "하단"
