from __future__ import annotations

from model.defect_info import Detection
from model.yolo_result import YoloResult
from vlm.response_parser import DEFAULT_OPERATOR_CHECK, sanitize_vlm_explanation


def make_result(confidence: float = 0.784) -> YoloResult:
    return YoloResult(
        image_path="sample.jpg",
        detections=[
            Detection(0, "open_circuit", confidence, 2711, 946, 2739, 979, location="middle right")
        ],
    )


def test_sanitize_replaces_no_check_required_for_ng() -> None:
    response = {
        "detections": [
            {"detection_id": 1, "visual_feature": "visible gap", "operator_check": "no check required"}
        ]
    }

    text = sanitize_vlm_explanation(str(response).replace("'", '"'), make_result())

    assert DEFAULT_OPERATOR_CHECK in text
    assert "no check required" not in text.lower()


def test_sanitize_keeps_yolo_ng_when_vlm_text_says_ok() -> None:
    text = sanitize_vlm_explanation("1. Final judgment: OK", make_result())

    assert "최종 판정: NG" in text
    assert "Final judgment: OK" not in text


def test_sanitize_keeps_yolo_ok_when_vlm_text_says_ng() -> None:
    result = YoloResult(image_path="sample.jpg", detections=[])

    text = sanitize_vlm_explanation("1. Final judgment: NG", result)

    assert "최종 판정: OK" in text
    assert "Final judgment: NG" not in text


def test_sanitize_empty_response_uses_safe_fallback() -> None:
    text = sanitize_vlm_explanation("   ", make_result())

    assert "최종 판정: NG" in text
    assert "탐지 수: 1" in text
    assert DEFAULT_OPERATOR_CHECK in text


def test_sanitize_uses_yolo_class_count_location_confidence_and_bbox() -> None:
    text = sanitize_vlm_explanation("normal draft", make_result(confidence=0.6696))

    assert "클래스: open_circuit" in text
    assert "탐지 수: 1" in text
    assert "위치: middle right" in text
    assert "신뢰도: 0.6696" in text
    assert "Bounding Box: (2711, 946, 2739, 979)" in text
    assert "우선 재검토 여부: 예" in text


def test_sanitize_json_response_formats_structured_text_with_yolo_authority() -> None:
    response = """
    {
      "final_judgment": "OK",
      "defect_classes": ["short"],
      "detection_count": 99,
      "detections": [
        {
          "detection_id": 1,
          "class_name": "short",
          "location": "elsewhere",
          "confidence": 0.1,
          "bounding_box": [0, 0, 1, 1],
          "visual_feature": "trace gap is visible",
          "uncertainty": "slight blur",
          "operator_check": "review crop",
          "priority_recheck": false
        }
      ],
      "overall_reason": "visual note",
      "final_operator_check": "confirm with original"
    }
    """

    text = sanitize_vlm_explanation(response, make_result())

    assert "최종 판정: NG" in text
    assert "클래스: open_circuit" in text
    assert "탐지 수: 1" in text
    assert "1. open_circuit" in text
    assert "위치: middle right" in text
    assert "신뢰도: 0.7840" in text
    assert "Bounding Box: (2711, 946, 2739, 979)" in text
    assert "trace gap is visible" in text
    assert "visual note" in text
    assert '"final_judgment"' not in text


def test_sanitize_uses_detection_id_when_vlm_order_is_changed() -> None:
    result = YoloResult(
        image_path="sample.jpg",
        detections=[
            Detection(0, "open_circuit", 0.8, 1, 2, 3, 4, location="first"),
            Detection(0, "open_circuit", 0.8, 5, 6, 7, 8, location="second"),
        ],
    )
    response = """
    ```json
    {
      "detections": [
        {"detection_id": "2", "visual_feature": "feature for second"},
        {"detection_id": "1", "visual_feature": "feature for first"}
      ]
    }
    ```
    """

    text = sanitize_vlm_explanation(response, result)

    assert text.index("feature for first") < text.index("feature for second")


def test_sanitize_missing_vlm_fields_uses_fallback_and_priority_recheck() -> None:
    response = '{"detections": [{"detection_id": 1}]}'

    text = sanitize_vlm_explanation(response, make_result())

    assert "crop 이미지에서 구체적인 시각적 특징" in text
    assert "우선 재검토 여부: 예" in text


def test_sanitize_generic_class_definition_visual_feature_uses_fallback() -> None:
    response = """
    {
      "detections": [
        {
          "detection_id": 1,
          "visual_feature": "broken or missing circuit pattern, discontinuity, connection state"
        }
      ]
    }
    """

    text = sanitize_vlm_explanation(response, make_result())

    assert "broken or missing circuit pattern" not in text
    assert "crop 이미지에서 구체적인 시각적 특징" in text
    assert "우선 재검토 여부: 예" in text


def test_sanitize_malformed_json_falls_back_without_raw_json() -> None:
    response = '{"detections": [{"detection_id": 1, "visual_feature": "x"}'

    text = sanitize_vlm_explanation(response, make_result())

    assert "최종 판정: NG" in text
    assert "탐지 수: 1" in text
    assert '"detections"' not in text
