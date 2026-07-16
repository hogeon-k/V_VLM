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
    has_class_conflict,
    has_language_warning,
    has_location_leak,
    has_summary_contradiction,
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
                    "visual_feature": "패턴 경계가 중간에서 불연속적으로 보입니다.",
                    "visibility": "clear",
                    "review_required": False,
                }
            ],
            "summary": "총 1개의 결함이 탐지되었으며, 1개는 시각적 특징이 명확합니다.",
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
    assert "패턴 경계가 중간에서 불연속적으로 보입니다." in result.formatted_response
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
        ("open_circuit.", "open_circuit"),
        ("open_circuit:", "open_circuit"),
        ("Open Circuit", "open_circuit"),
        ("open-circuit defect", "open_circuit"),
        ("short:", "short"),
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
    assert visual_feature_quality(DEFAULT_VISUAL_FEATURE, "short") == "acceptable"
    assert visual_feature_quality("No clear visual characteristic could be confirmed.", "short") == "fallback"


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
    assert quality.semantic_warning_count == 2


@pytest.mark.parametrize(
    ("class_name", "visual_feature"),
    [
        ("short", "missing_hole"),
        ("short", "홀이 보이지 않습니다."),
        ("short", "회로가 끊어짐이 보입니다."),
        ("open_circuit", "두 패턴이 연결됨처럼 보입니다."),
        ("open_circuit", "누락된 홀처럼 보입니다."),
        ("missing_hole", "패턴이 끊어짐처럼 보입니다."),
        ("missing_hole", "패턴 연결 형태가 보입니다."),
    ],
)
def test_class_conflict_detects_other_class_language(
    class_name: str,
    visual_feature: str,
) -> None:
    assert has_class_conflict(visual_feature, class_name) is True


def test_class_conflict_allows_matching_visual_description() -> None:
    assert has_class_conflict("두 도전성 패턴 사이가 가느다란 패턴으로 연결된 것처럼 보입니다.", "short") is False


@pytest.mark.parametrize(
    "visual_feature",
    [
        "상단 오른쪽에 누락된 영역이 보입니다.",
        "bottom right corner에 결함이 보입니다.",
        "중앙에 경계 차이가 보입니다.",
    ],
)
def test_location_leak_detects_position_words(visual_feature: str) -> None:
    assert has_location_leak(visual_feature) is True


@pytest.mark.parametrize(
    "visual_feature",
    [
        "A visible gap interrupts the trace.",
        "패턴 경계가 清晰하게 보입니다.",
        "패턴 edge is visibly broken.",
    ],
)
def test_language_warning_detects_english_or_chinese_text(visual_feature: str) -> None:
    assert has_language_warning(visual_feature) is True


@pytest.mark.parametrize(
    "visual_feature",
    [
        "두 도전성 패턴 사이가 가느다란 패턴으로 연결된 것처럼 보입니다.",
        DEFAULT_VISUAL_FEATURE,
    ],
)
def test_language_warning_allows_korean_visual_descriptions(visual_feature: str) -> None:
    assert has_language_warning(visual_feature) is False


def test_evaluate_response_quality_records_semantic_warning_detection_ids() -> None:
    raw = json.dumps(
        {
            "final_judgment": "NG",
            "detections": [
                {
                    "detection_id": 1,
                    "visual_feature": "open_circuit:",
                    "visibility": "clear",
                    "review_required": False,
                },
                {
                    "detection_id": 2,
                    "visual_feature": "상단 오른쪽에서 홀이 보이지 않습니다.",
                    "visibility": "clear",
                    "review_required": False,
                },
                {
                    "detection_id": 3,
                    "visual_feature": "A visible gap interrupts the trace.",
                    "visibility": "clear",
                    "review_required": False,
                },
            ],
            "summary": "총 3개의 결함이 탐지되었으며, 3개는 시각적 특징이 명확합니다.",
        }
    )
    parsed = parse_vlm_response(raw, expected_detection_count=3)
    yolo_result = YoloResult(
        image_path="sample.jpg",
        detections=[
            Detection(0, "open_circuit", 0.9, 1, 2, 3, 4),
            Detection(1, "short", 0.8, 5, 6, 7, 8),
            Detection(2, "missing_hole", 0.7, 9, 10, 11, 12),
        ],
    )

    quality = evaluate_response_quality(parsed, yolo_result)

    assert quality.quality_status == "warning"
    assert quality.class_name_only_detection_ids == (1,)
    assert quality.class_conflict_detection_ids == (2,)
    assert quality.location_leak_detection_ids == (2,)
    assert quality.language_warning_detection_ids == (3,)
    assert quality.semantic_warning_count == 4


def test_parser_preserves_raw_response_and_detection_order_with_quality_warning() -> None:
    raw = json.dumps(
        {
            "final_judgment": "NG",
            "detections": [
                {
                    "detection_id": 1,
                    "visual_feature": "short:",
                    "visibility": "clear",
                    "review_required": False,
                }
            ],
            "summary": "총 1개의 결함이 탐지되었습니다.",
        }
    )
    result = VlmResponseParser().parse_response(
        raw,
        YoloResult(
            image_path="sample.jpg",
            detections=[Detection(0, "short", 0.9, 1, 2, 3, 4)],
        ),
    )

    assert result.parse_success is True
    assert result.raw_response == raw
    assert result.parsed_response is not None
    assert [detection.detection_id for detection in result.parsed_response.detections] == [1]
    assert result.parsed_response.detections[0].visual_feature == "short:"
    assert result.quality_info.class_name_only_detection_ids == (1,)


@pytest.mark.parametrize(
    ("class_name", "visual_feature", "safe_phrase"),
    [
        (
            "short",
            "회로 패턴이 중간에서 끊겨 보이는 구간이 있습니다.",
            "두 도전성 패턴 사이의 비정상적인 연결 여부를 명확히 확인하기 어렵습니다.",
        ),
        (
            "open_circuit",
            "두 도전성 패턴 사이가 가느다란 패턴으로 연결되어 있습니다.",
            "회로 패턴의 단절 여부를 명확히 확인하기 어렵습니다.",
        ),
        (
            "missing_hole",
            "두 회로 패턴 사이가 연결되어 있습니다.",
            "원형 홀의 누락 여부를 명확히 확인하기 어렵습니다.",
        ),
    ],
)
def test_parser_corrects_class_conflicting_visual_feature_without_fallback(
    class_name: str,
    visual_feature: str,
    safe_phrase: str,
) -> None:
    raw = json.dumps(
        {
            "final_judgment": "NG",
            "detections": [
                {
                    "detection_id": 1,
                    "visual_feature": visual_feature,
                    "visibility": "clear",
                    "review_required": False,
                }
            ],
            "summary": "총 1개의 결함이 탐지되었으며, 1개는 시각적 특징이 명확합니다.",
        }
    )
    yolo_result = YoloResult(
        image_path="sample.jpg",
        detections=[Detection(0, class_name, 0.9, 1, 2, 3, 4)],
    )

    result = VlmResponseParser().parse_response(raw, yolo_result)

    assert result.parse_success is True
    assert result.fallback_used is False
    assert result.quality_info.quality_status == "warning"
    assert result.quality_info.class_conflict_detection_ids == (1,)
    assert result.quality_info.summary_contradiction is True
    assert result.parsed_response is not None
    corrected = result.parsed_response.detections[0]
    assert corrected.visual_feature == f"확대 이미지에서 {safe_phrase}"
    assert corrected.visibility == "unclear"
    assert corrected.review_required is True
    assert f"1. {class_name}" in result.formatted_response
    assert safe_phrase in result.formatted_response
    assert "추가 확인: 필요" in result.formatted_response


def test_short_connection_description_is_not_marked_as_class_conflict() -> None:
    visual_feature = "두 도전성 패턴 사이가 가느다란 연결부로 이어져 보입니다."

    assert has_class_conflict(visual_feature, "short") is False


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


def test_summary_contradiction_detects_detection_count_mismatch() -> None:
    parsed = parse_vlm_response(
        json.dumps(
            {
                "final_judgment": "NG",
                "detections": [
                    {
                        "detection_id": 1,
                        "visual_feature": "패턴 경계가 불연속적으로 보입니다.",
                        "visibility": "clear",
                        "review_required": False,
                    },
                    {
                        "detection_id": 2,
                        "visual_feature": DEFAULT_VISUAL_FEATURE,
                        "visibility": "unclear",
                        "review_required": True,
                    },
                    {
                        "detection_id": 3,
                        "visual_feature": "패턴 경계가 불연속적으로 보입니다.",
                        "visibility": "clear",
                        "review_required": False,
                    },
                ],
                "summary": "두 결함이 탐지되었습니다.",
            }
        ),
        expected_detection_count=3,
    )

    assert has_summary_contradiction(parsed, YoloResult("sample.jpg", [])) is True


def test_summary_contradiction_allows_matching_clear_unclear_counts() -> None:
    parsed = parse_vlm_response(
        json.dumps(
            {
                "final_judgment": "NG",
                "detections": [
                    {
                        "detection_id": 1,
                        "visual_feature": "패턴 경계가 불연속적으로 보입니다.",
                        "visibility": "clear",
                        "review_required": False,
                    },
                    {
                        "detection_id": 2,
                        "visual_feature": "패턴 경계가 불연속적으로 보입니다.",
                        "visibility": "clear",
                        "review_required": False,
                    },
                    {
                        "detection_id": 3,
                        "visual_feature": DEFAULT_VISUAL_FEATURE,
                        "visibility": "unclear",
                        "review_required": True,
                    },
                ],
                "summary": "총 3개의 결함이 탐지되었으며, 2개는 시각적 특징이 명확하고 1개는 추가 확인이 필요합니다.",
            }
        ),
        expected_detection_count=3,
    )

    assert has_summary_contradiction(parsed, YoloResult("sample.jpg", [])) is False


@pytest.mark.parametrize(
    "summary",
    [
        "총 1개의 결함이 탐지되었으며 open_circuit이 보입니다.",
        "총 1개의 결함이 탐지되었으며 신뢰도 0.91입니다.",
        "총 1개의 결함이 탐지되었으며 bbox (1, 2, 3, 4)를 확인했습니다.",
    ],
)
def test_summary_contradiction_detects_new_class_confidence_or_bbox(summary: str) -> None:
    parsed = parse_vlm_response(
        json.dumps(
            {
                "final_judgment": "NG",
                "detections": [
                    {
                        "detection_id": 1,
                        "visual_feature": "패턴 경계가 불연속적으로 보입니다.",
                        "visibility": "clear",
                        "review_required": False,
                    }
                ],
                "summary": summary,
            }
        ),
        expected_detection_count=1,
    )

    assert has_summary_contradiction(parsed, YoloResult("sample.jpg", [])) is True


def test_formatting_is_deterministic_and_yolo_authoritative() -> None:
    parsed = parse_vlm_response(valid_raw_response(), expected_detection_count=1)
    yolo_result = make_result(confidence=0.6696)

    first = format_parsed_vlm_response(parsed, yolo_result)
    second = format_parsed_vlm_response(parsed, yolo_result)

    assert first == second
    assert "신뢰도: 0.6696" in first
    assert "바운딩 박스: (2711, 946, 2739, 979)" in first
    assert "최종 판정: NG" in first


def test_formatting_replaces_contradictory_vlm_summary_with_detection_counts() -> None:
    parsed = parse_vlm_response(
        json.dumps(
            {
                "final_judgment": "NG",
                "detections": [
                    {
                        "detection_id": 1,
                        "visual_feature": "패턴 경계가 흐리게 보입니다.",
                        "visibility": "clear",
                        "review_required": False,
                    },
                    {
                        "detection_id": 2,
                        "visual_feature": DEFAULT_VISUAL_FEATURE,
                        "visibility": "unclear",
                        "review_required": True,
                    },
                    {
                        "detection_id": 3,
                        "visual_feature": DEFAULT_VISUAL_FEATURE,
                        "visibility": "unclear",
                        "review_required": True,
                    },
                ],
                "summary": "총 3개의 결함이 탐지되었으며, 2개는 시각적 특징이 명확하고 1개는 추가 확인이 필요합니다.",
            }
        ),
        expected_detection_count=3,
    )
    yolo_result = YoloResult(
        "sample.jpg",
        [
            Detection(0, "short", 0.9, 1, 2, 3, 4),
            Detection(0, "short", 0.8, 5, 6, 7, 8),
            Detection(0, "short", 0.7, 9, 10, 11, 12),
        ],
    )

    formatted = format_parsed_vlm_response(parsed, yolo_result)

    assert "1개는 시각적 특징이 명확하고 2개는 추가 확인이 필요합니다." in formatted
    assert "2개는 시각적 특징이 명확하고 1개는 추가 확인이 필요합니다." not in formatted


def test_sanitize_vlm_explanation_compatibility_wrapper() -> None:
    text = sanitize_vlm_explanation("invalid", make_result())

    assert "최종 판정: NG" in text
    assert DEFAULT_VISUAL_FEATURE in text
