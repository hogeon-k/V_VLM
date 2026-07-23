from __future__ import annotations

import numpy as np
import pytest

from model.defect_info import Detection
from service.onnx_detector import (
    LetterboxInfo,
    bbox_iou,
    class_aware_nms,
    detection_to_dict,
    letterbox,
    postprocess_output,
    restore_boxes_to_original,
    validate_onnx_output,
    xywh_to_xyxy,
)


def test_letterbox_result_shape() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    padded, info = letterbox(image, new_shape=960)

    assert padded.shape == (960, 960, 3)
    assert info.original_shape == (100, 200)
    assert info.new_unpad == (960, 480)
    assert info.pad == (0, 240)


def test_letterbox_coordinate_restore() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    _, info = letterbox(image, new_shape=960)
    original_box = np.array([[50, 20, 150, 80]], dtype=np.float32)
    letterboxed_box = original_box.copy()
    letterboxed_box[:, [0, 2]] = letterboxed_box[:, [0, 2]] * info.ratio[0] + info.pad[0]
    letterboxed_box[:, [1, 3]] = letterboxed_box[:, [1, 3]] * info.ratio[1] + info.pad[1]

    restored = restore_boxes_to_original(letterboxed_box, info)

    np.testing.assert_allclose(restored, original_box, atol=1e-4)


def test_xywh_to_xyxy() -> None:
    boxes = np.array([[100, 50, 20, 10]], dtype=np.float32)

    converted = xywh_to_xyxy(boxes)

    np.testing.assert_allclose(converted, np.array([[90, 45, 110, 55]], dtype=np.float32))


def test_bbox_iou() -> None:
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert bbox_iou([0, 0, 10, 10], [10, 10, 20, 20]) == pytest.approx(0.0)
    assert bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)


def test_class_aware_nms_keeps_different_classes() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, 1], dtype=np.int32)

    keep = class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)

    assert keep == [0, 1]


def test_class_aware_nms_suppresses_same_class_overlap() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, 0], dtype=np.int32)

    keep = class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)

    assert keep == [0]


def test_postprocess_empty_detection() -> None:
    output = np.zeros((1, 7, 2), dtype=np.float32)
    info = LetterboxInfo(
        original_shape=(100, 100),
        resized_shape=(960, 960),
        ratio=(9.6, 9.6),
        pad=(0, 0),
        new_unpad=(960, 960),
    )

    detections = postprocess_output(output, info, conf_threshold=0.15, iou_threshold=0.5)

    assert detections == []


def test_validate_onnx_output_shape() -> None:
    output = np.zeros((1, 7, 3), dtype=np.float32)

    validated = validate_onnx_output(output)

    assert validated.shape == (3, 7)


def test_validate_onnx_output_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        validate_onnx_output(np.zeros((7, 3), dtype=np.float32))


def test_detection_to_dict_structure() -> None:
    detection = Detection(1, "short", 0.8, 1, 2, 3, 4)

    data = detection_to_dict(detection)

    assert data == {
        "class_id": 1,
        "class_name": "short",
        "confidence": 0.8,
        "bbox": [1, 2, 3, 4],
    }
