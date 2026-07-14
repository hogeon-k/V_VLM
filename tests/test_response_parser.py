from __future__ import annotations

import json

import pytest

from model.defect_info import Detection
from model.yolo_result import YoloResult
from vlm.response_parser import (
    DEFAULT_VISUAL_FEATURE,
    FALLBACK_SUMMARY,
    VlmResponseParser,
    evaluate_response_quality,
    format_parsed_vlm_response,
    has_summary_visibility_contradiction,
    is_class_name_only_visual_feature,
    parse_vlm_response,
    sanitize_vlm_explanation,
    visual_feature_quality,
)


def make_result(confidence: float = 0.784) -> YoloResult:
    return YoloResult(
        image_path="sample.jpg",
        detections=[
            Detection(0, "open_circuit", confidence, 2711, 946, 2739, 979, location="middle right")
        ],
    )


def valid_raw_response() -> str:
    return json.dumps(
        {
            "final_judgment": "NG",
            "detections": [
                {
                    "detection_id": 1,
                    "visual_feature": "A visible gap interrupts the trace.",
                    "visibility": "clear",
                    "review_required": False,
                }
            ],
            "summary": "One YOLO detection has a visible trace interruption.",
        }
    )


def test_parse_valid_json_succeeds() -> None:
    parsed = parse_vlm_response(valid_raw_response(), expected_detection_count=1)

    assert parsed.final_judgment == "NG"
    assert parsed.detections[0].detection_id == 1
    assert parsed.detections[0].visibility == "clear"


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_vlm_response("{", expected_detection_count=1)


def test_parse_missing_required_field_raises() -> None:
    raw = '{"final_judgment": "NG", "detections": []}'

    with pytest.raises(ValueError, match="summary"):
        parse_vlm_response(raw, expected_detection_count=0)


def test_parse_rejects_additional_properties() -> None:
    data = json.loads(valid_raw_response())
    data["extra"] = "nope"

    with pytest.raises(ValueError, match="Unexpected field"):
        parse_vlm_response(json.dumps(data), expected_detection_count=1)


def test_parse_detection_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Detection count mismatch"):
        parse_vlm_response(valid_raw_response(), expected_detection_count=2)


def test_parse_detection_id_mismatch_raises() -> None:
    data = json.loads(valid_raw_response())
    data["detections"][0]["detection_id"] = 2

    with pytest.raises(ValueError, match="Detection ID mismatch"):
        parse_vlm_response(json.dumps(data), expected_detection_count=1)


def test_parse_reordered_detection_ids_raise() -> None:
    data = json.loads(valid_raw_response())
    data["detections"].append(
        {
            "detection_id": 1,
            "visual_feature": "second",
            "visibility": "unclear",
            "review_required": True,
        }
    )
    data["detections"][0]["detection_id"] = 2

    with pytest.raises(ValueError, match="Detection ID mismatch"):
        parse_vlm_response(json.dumps(data), expected_detection_count=2)


def test_parse_empty_response_raises() -> None:
    with pytest.raises(ValueError, match="Empty VLM response"):
        parse_vlm_response("  ", expected_detection_count=1)


def test_parser_result_records_success_metadata_and_formats_text() -> None:
    raw = valid_raw_response()
    result = VlmResponseParser().parse_response(raw, make_result())

    assert result.parse_success is True
    assert result.fallback_used is False
    assert result.parse_error == ""
    assert result.raw_response == raw
    assert "최종 판정: NG" in result.formatted_response
    assert "1. open_circuit" in result.formatted_response
    assert "신뢰도: 0.7840" in result.formatted_response
    assert "위치: middle right" in result.formatted_response
    assert "A visible gap interrupts the trace." in result.formatted_response
    assert "가시성: 명확함" in result.formatted_response
    assert "추가 확인: 불필요" in result.formatted_response
    assert result.quality_info.quality_status == "acceptable"


def test_parser_result_records_failure_metadata_and_retains_raw_response() -> None:
    raw = "not json"
    result = VlmResponseParser().parse_response(raw, make_result())

    assert result.parse_success is False
    assert result.fallback_used is True
    assert result.parse_error.startswith("Invalid JSON")
    assert result.raw_response == raw
    assert FALLBACK_SUMMARY in result.formatted_response
    assert raw not in result.formatted_response
    assert result.quality_info.quality_status == "not_evaluated"


@pytest.mark.parametrize(
    ("visual_feature", "class_name"),
    [
        ("short", "short"),
        ("Short", "short"),
        ("short defect", "short"),
        ("defect short", "short"),
        ("open_circuit", "open_circuit"),
        ("Open Circuit", "open_circuit"),
        ("open-circuit defect", "open_circuit"),
        ("missing_hole", "missing_hole"),
        ("missing hole defect", "missing_hole"),
    ],
)
def test_class_name_only_visual_feature_normalizes_variants(
    visual_feature: str,
    class_name: str,
) -> None:
    assert is_class_name_only_visual_feature(visual_feature, class_name) is True


@pytest.mark.parametrize(
    "visual_feature",
    [
        "a thin conductive bridge connects adjacent traces",
        "a visible gap interrupts the copper line",
        "a circular pad lacks the expected drilled opening",
        "a short conductive bridge connects adjacent traces",
    ],
)
def test_visual_feature_quality_allows_observable_shape_descriptions(
    visual_feature: str,
) -> None:
    assert visual_feature_quality(visual_feature, "short") == "acceptable"


def test_visual_feature_quality_handles_empty_and_fallback() -> None:
    assert visual_feature_quality("", "short") == "empty"
    assert visual_feature_quality(DEFAULT_VISUAL_FEATURE, "short") == "fallback"


def test_evaluate_response_quality_warns_for_class_name_only() -> None:
    raw = json.dumps(
        {
            "final_judgment": "NG",
            "detections": [
                {
                    "detection_id": 1,
                    "visual_feature": "open circuit defect",
                    "visibility": "clear",
                    "review_required": False,
                }
            ],
            "summary": "One defect is visible.",
        }
    )
    parsed = parse_vlm_response(raw, expected_detection_count=1)

    quality = evaluate_response_quality(parsed, make_result())

    assert quality.quality_status == "warning"
    assert quality.class_name_only_count == 1
    assert quality.class_name_only_detection_ids == (1,)
    assert quality.semantic_warning_count == 1


def test_summary_visibility_contradiction_detects_explicit_all_clear() -> None:
    raw = json.dumps(
        {
            "final_judgment": "NG",
            "detections": [
                {
                    "detection_id": 1,
                    "visual_feature": "A gap is hard to confirm.",
                    "visibility": "unclear",
                    "review_required": True,
                }
            ],
            "summary": "All defects are clearly visible.",
        }
    )
    parsed = parse_vlm_response(raw, expected_detection_count=1)

    assert has_summary_visibility_contradiction(parsed) is True


def test_summary_visibility_contradiction_allows_mixed_ambiguous_summary() -> None:
    raw = json.dumps(
        {
            "final_judgment": "NG",
            "detections": [
                {
                    "detection_id": 1,
                    "visual_feature": "A gap is hard to confirm.",
                    "visibility": "unclear",
                    "review_required": True,
                }
            ],
            "summary": "One defect remains ambiguous.",
        }
    )
    parsed = parse_vlm_response(raw, expected_detection_count=1)

    assert has_summary_visibility_contradiction(parsed) is False


def test_formatting_is_deterministic_and_yolo_authoritative() -> None:
    parsed = parse_vlm_response(valid_raw_response(), expected_detection_count=1)
    yolo_result = make_result(confidence=0.6696)

    first = format_parsed_vlm_response(parsed, yolo_result)
    second = format_parsed_vlm_response(parsed, yolo_result)

    assert first == second
    assert "신뢰도: 0.6696" in first
    assert "바운딩 박스: (2711, 946, 2739, 979)" in first
    assert "최종 판정: NG" in first


def test_sanitize_vlm_explanation_compatibility_wrapper() -> None:
    text = sanitize_vlm_explanation("invalid", make_result())

    assert "최종 판정: NG" in text
    assert DEFAULT_VISUAL_FEATURE in text
