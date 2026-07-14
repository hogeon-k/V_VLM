from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from model.yolo_result import YoloResult

ENGLISH_DEFAULT_VISUAL_FEATURE = "No clear visual characteristic could be confirmed."
DEFAULT_VISUAL_FEATURE = "명확한 시각적 특징을 확인하지 못했습니다."
FALLBACK_SUMMARY = (
    "VLM 설명 생성에 실패하여 YOLO 탐지 결과를 기준으로 표시합니다."
)


@dataclass(frozen=True)
class ParsedVlmDetection:
    detection_id: int
    visual_feature: str
    visibility: str
    review_required: bool


@dataclass(frozen=True)
class ParsedVlmResponse:
    final_judgment: str
    detections: list[ParsedVlmDetection]
    summary: str
    raw_data: dict[str, Any]


@dataclass(frozen=True)
class VlmQualityInfo:
    quality_status: str = "not_evaluated"
    class_name_only_count: int = 0
    class_name_only_detection_ids: tuple[int, ...] = ()
    summary_contradiction: bool = False
    semantic_warning_count: int = 0


@dataclass(frozen=True)
class VlmParseResult:
    raw_response: str
    parse_success: bool
    parse_error: str
    fallback_used: bool
    parsed_response: ParsedVlmResponse | None
    formatted_response: str
    quality_info: VlmQualityInfo = field(default_factory=VlmQualityInfo)


class VlmResponseParser:
    def parse_response(self, raw_response: str, yolo_result: YoloResult) -> VlmParseResult:
        """Parse structured JSON and fall back to YOLO-authoritative text on failure."""
        try:
            parsed_response = parse_vlm_response(raw_response, yolo_result.defect_count)
            quality_info = evaluate_response_quality(parsed_response, yolo_result)
            formatted = format_parsed_vlm_response(parsed_response, yolo_result)
            return VlmParseResult(
                raw_response=raw_response,
                parse_success=True,
                parse_error="",
                fallback_used=False,
                parsed_response=parsed_response,
                formatted_response=formatted,
                quality_info=quality_info,
            )
        except ValueError as exc:
            return VlmParseResult(
                raw_response=raw_response,
                parse_success=False,
                parse_error=str(exc),
                fallback_used=True,
                parsed_response=None,
                formatted_response=format_yolo_fallback_response(yolo_result),
                quality_info=VlmQualityInfo(),
            )

    def parse_description(self, response_text: str, yolo_result: YoloResult | None = None) -> str:
        """Backward-compatible wrapper returning only the user-facing text."""
        if yolo_result is None:
            return response_text.strip()
        return self.parse_response(response_text, yolo_result).formatted_response


def parse_vlm_response(raw_response: str, expected_detection_count: int) -> ParsedVlmResponse:
    """Validate the exact JSON structure expected from the VLM."""
    if not raw_response or not raw_response.strip():
        raise ValueError("Empty VLM response")

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ValueError("Unexpected response structure: root must be an object")

    _reject_additional_properties(data, {"final_judgment", "detections", "summary"}, "root")

    final_judgment = data.get("final_judgment")
    if final_judgment not in {"OK", "NG"}:
        raise ValueError(f"Invalid final_judgment: {final_judgment}")

    detections = data.get("detections")
    if not isinstance(detections, list):
        raise ValueError("Missing or invalid detections field")

    if len(detections) != expected_detection_count:
        raise ValueError(
            "Detection count mismatch: "
            f"expected={expected_detection_count}, actual={len(detections)}"
        )

    parsed_detections: list[ParsedVlmDetection] = []
    for index, item in enumerate(detections, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Unexpected detection structure at index {index}")
        _reject_additional_properties(
            item,
            {"detection_id", "visual_feature", "visibility", "review_required"},
            f"detection {index}",
        )
        missing = [
            key
            for key in ("detection_id", "visual_feature", "visibility", "review_required")
            if key not in item
        ]
        if missing:
            raise ValueError(f"Missing required field in detection {index}: {missing[0]}")

        detection_id = item["detection_id"]
        if isinstance(detection_id, bool) or not isinstance(detection_id, int):
            raise ValueError(f"Invalid detection_id at index {index}: {detection_id}")

        visual_feature = item["visual_feature"]
        if not isinstance(visual_feature, str):
            raise ValueError(f"Invalid visual_feature for detection {detection_id}")

        visibility = item["visibility"]
        if visibility not in {"clear", "unclear"}:
            raise ValueError(f"Invalid visibility for detection {detection_id}: {visibility}")

        review_required = item["review_required"]
        if not isinstance(review_required, bool):
            raise ValueError(f"Invalid review_required for detection {detection_id}")

        parsed_detections.append(
            ParsedVlmDetection(
                detection_id=detection_id,
                visual_feature=visual_feature.strip() or DEFAULT_VISUAL_FEATURE,
                visibility=visibility,
                review_required=review_required,
            )
        )

    expected_ids = list(range(1, expected_detection_count + 1))
    actual_ids = [detection.detection_id for detection in parsed_detections]
    if actual_ids != expected_ids:
        raise ValueError(f"Detection ID mismatch: expected={expected_ids}, actual={actual_ids}")

    summary = data.get("summary")
    if not isinstance(summary, str):
        raise ValueError("Missing or invalid summary field")

    return ParsedVlmResponse(
        final_judgment=final_judgment,
        detections=parsed_detections,
        summary=summary.strip(),
        raw_data=data,
    )


def evaluate_response_quality(
    parsed_response: ParsedVlmResponse,
    yolo_result: YoloResult,
) -> VlmQualityInfo:
    """Evaluate explanation quality without changing parse success/fallback behavior."""
    class_name_only_ids: list[int] = []
    fallback_or_empty_count = 0
    for yolo_detection, vlm_detection in zip(
        yolo_result.detections,
        parsed_response.detections,
        strict=True,
    ):
        quality = visual_feature_quality(
            vlm_detection.visual_feature,
            yolo_detection.class_name,
        )
        if quality == "class_name_only":
            class_name_only_ids.append(vlm_detection.detection_id)
        elif quality in {"empty", "fallback"}:
            fallback_or_empty_count += 1
    summary_contradiction = has_summary_visibility_contradiction(parsed_response)
    semantic_warning_count = (
        len(class_name_only_ids) + fallback_or_empty_count + int(summary_contradiction)
    )
    quality_status = "warning" if semantic_warning_count else "acceptable"
    return VlmQualityInfo(
        quality_status=quality_status,
        class_name_only_count=len(class_name_only_ids),
        class_name_only_detection_ids=tuple(class_name_only_ids),
        summary_contradiction=summary_contradiction,
        semantic_warning_count=semantic_warning_count,
    )


def is_class_name_only_visual_feature(visual_feature: str, class_name: str) -> bool:
    """Return true when visual_feature is only a normalized class label."""
    normalized_feature = _normalize_class_label(visual_feature)
    if not normalized_feature:
        return False
    normalized_class = _normalize_class_label(class_name)
    class_only_forms = {
        normalized_class,
        f"{normalized_class} defect",
        f"defect {normalized_class}",
    }
    return normalized_feature in class_only_forms


def visual_feature_quality(visual_feature: str, class_name: str) -> str:
    """Classify one visual_feature quality with a small stable vocabulary."""
    if not visual_feature or not visual_feature.strip():
        return "empty"
    if visual_feature.strip() in {DEFAULT_VISUAL_FEATURE, ENGLISH_DEFAULT_VISUAL_FEATURE}:
        return "fallback"
    if is_class_name_only_visual_feature(visual_feature, class_name):
        return "class_name_only"
    return "acceptable"


def has_summary_visibility_contradiction(parsed_response: ParsedVlmResponse) -> bool:
    """Detect explicit 'all clear' summaries when any detection is unclear."""
    if not any(detection.visibility == "unclear" for detection in parsed_response.detections):
        return False
    normalized_summary = re.sub(r"\s+", " ", parsed_response.summary.lower()).strip()
    contradiction_patterns = (
        "all defects are clearly visible",
        "all detections are clearly visible",
        "all defects are clear",
        "every defect is clearly visible",
        "모든 불량이 명확하게 보입니다",
        "모든 탐지 영역이 명확합니다",
        "모든 불량이 선명하게 확인됩니다",
    )
    return any(pattern in normalized_summary for pattern in contradiction_patterns)


def format_parsed_vlm_response(
    parsed_response: ParsedVlmResponse,
    yolo_result: YoloResult,
) -> str:
    """Format parsed VLM observations deterministically using YOLO metadata."""
    yolo_final_judgment = "NG" if yolo_result.is_ng else "OK"
    lines = [
        f"최종 판정: {yolo_final_judgment}",
        "",
        "탐지 요약:",
        f"- 탐지된 불량 수: {len(yolo_result.detections)}개",
        "",
        "불량 상세 정보:",
    ]

    for index, (yolo_detection, vlm_detection) in enumerate(
        zip(yolo_result.detections, parsed_response.detections, strict=True),
        start=1,
    ):
        lines.extend(
            [
                "",
                f"{index}. {yolo_detection.class_name}",
                f"   - 신뢰도: {yolo_detection.confidence:.4f}",
                f"   - 위치: {yolo_detection.location or '위치 정보 없음'}",
                (
                    "   - 바운딩 박스: "
                    f"({yolo_detection.x1}, {yolo_detection.y1}, "
                    f"{yolo_detection.x2}, {yolo_detection.y2})"
                ),
                f"   - 시각적 특징: {vlm_detection.visual_feature}",
                f"   - 가시성: {_format_visibility(vlm_detection.visibility)}",
                f"   - 추가 확인: {_format_review_required(vlm_detection.review_required)}",
            ]
        )

    lines.extend(["", "종합 설명:", parsed_response.summary])
    return "\n".join(lines)


def format_yolo_fallback_response(yolo_result: YoloResult) -> str:
    """Create deterministic safe text when the VLM response cannot be parsed."""
    final_judgment = "NG" if yolo_result.is_ng else "OK"
    lines = [
        f"최종 판정: {final_judgment}",
        "",
        "탐지 요약:",
        f"- 탐지된 불량 수: {len(yolo_result.detections)}개",
        "",
        "불량 상세 정보:",
    ]
    if not yolo_result.detections:
        lines.append("")
        lines.append("(없음)")
    for index, detection in enumerate(yolo_result.detections, start=1):
        lines.extend(
            [
                "",
                f"{index}. {detection.class_name}",
                f"   - 신뢰도: {detection.confidence:.4f}",
                f"   - 위치: {detection.location or '위치 정보 없음'}",
                (
                    "   - 바운딩 박스: "
                    f"({detection.x1}, {detection.y1}, {detection.x2}, {detection.y2})"
                ),
                f"   - 시각적 특징: {DEFAULT_VISUAL_FEATURE}",
                "   - 가시성: 불명확함",
                "   - 추가 확인: 필요",
            ]
        )
    lines.extend(["", "종합 설명:", FALLBACK_SUMMARY])
    return "\n".join(lines)


def sanitize_vlm_explanation(response_text: str, yolo_result: YoloResult) -> str:
    """Backward-compatible function name for existing callers."""
    return VlmResponseParser().parse_description(response_text, yolo_result)


def _reject_additional_properties(
    data: dict[str, Any],
    allowed_keys: set[str],
    location: str,
) -> None:
    unexpected = sorted(set(data) - allowed_keys)
    if unexpected:
        raise ValueError(f"Unexpected field in {location}: {unexpected[0]}")


def _normalize_class_label(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _format_visibility(value: str) -> str:
    if value == "clear":
        return "명확함"
    if value == "unclear":
        return "불명확함"
    return value


def _format_review_required(value: bool) -> str:
    return "필요" if value else "불필요"
