from __future__ import annotations

import pytest

from service.detection_location import calculate_detection_location


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((10, 10, 20, 20), "상단 좌측"),
        ((140, 10, 160, 20), "상단 중앙"),
        ((280, 10, 290, 20), "상단 오른쪽"),
        ((10, 140, 20, 160), "중단 좌측"),
        ((140, 140, 160, 160), "중단 중앙"),
        ((280, 140, 290, 160), "중단 오른쪽"),
        ((10, 280, 20, 290), "하단 좌측"),
        ((140, 280, 160, 290), "하단 중앙"),
        ((280, 280, 290, 290), "하단 오른쪽"),
    ],
)
def test_calculate_detection_location_all_regions(box, expected) -> None:
    assert calculate_detection_location(box, 300, 300) == expected


@pytest.mark.parametrize(("width", "height"), [(0, 100), (100, 0), (-1, 100), (100, -1)])
def test_calculate_detection_location_rejects_invalid_image_size(width, height) -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        calculate_detection_location((0, 0, 10, 10), width, height)


@pytest.mark.parametrize("box", [(20, 10, 10, 30), (10, 30, 20, 10)])
def test_calculate_detection_location_rejects_wrong_coordinate_order(box) -> None:
    with pytest.raises(ValueError, match="x1 <= x2 and y1 <= y2"):
        calculate_detection_location(box, 300, 300)


def test_calculate_detection_location_clamps_boxes_outside_image() -> None:
    assert calculate_detection_location((-20, -10, 20, 30), 300, 300) == "상단 좌측"
    assert calculate_detection_location((280, 280, 340, 360), 300, 300) == "하단 오른쪽"


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((100, 10, 100, 10), "상단 중앙"),
        ((200, 10, 200, 10), "상단 오른쪽"),
        ((10, 100, 10, 100), "중단 좌측"),
        ((10, 200, 10, 200), "하단 좌측"),
    ],
)
def test_calculate_detection_location_boundary_values(box, expected) -> None:
    assert calculate_detection_location(box, 300, 300) == expected
