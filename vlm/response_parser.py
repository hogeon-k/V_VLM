from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any

from model.yolo_result import YoloResult

ENGLISH_DEFAULT_VISUAL_FEATURE = "No clear visual characteristic could be confirmed."
DEFAULT_VISUAL_FEATURE = "확대 이미지에서 결함 영역이 작거나 불명확하여 구체적인 시각적 특징을 확인하기 어렵습니다."
FALLBACK_SUMMARY = (
    "VLM 설명 생성에 실패하여 YOLO 탐지 결과를 기준으로 표시합니다."
)
SAFE_VISUAL_FEATURE_BY_CLASS = {
    "short": "확대 이미지에서 두 도전성 패턴 사이의 비정상적인 연결 여부를 명확히 확인하기 어렵습니다.",
    "open circuit": "확대 이미지에서 회로 패턴의 단절 여부를 명확히 확인하기 어렵습니다.",
    "missing hole": "확대 이미지에서 원형 홀의 누락 여부를 명확히 확인하기 어렵습니다.",
}

CLASS_CONFLICT_TERMS = {
    "short": (
        "open_circuit",
        "open circuit",
        "missing_hole",
        "missing hole",
        "단선",
        "누락된 홀",
        "홀이 보이지",
        "홀 형태가 보이지",
        "회로가 끊어",
        "패턴이 끊",
        "끊겨",
        "끊김",
        "끊어진",
        "단절",
        "절단",
        "분리",
        "이어지지 않",
    ),
    "open circuit": (
        "short",
        "short circuit",
        "missing_hole",
        "missing hole",
        "단락",
        "연결된 패턴",
        "두 패턴이 연결",
        "두 도전성 패턴 사이",
        "가느다란 패턴으로 연결",
        "서로 연결",
        "비정상적으로 이어",
        "브리지",
        "붙어 있",
        "누락된 홀",
        "홀이 보이지",
    ),
    "missing hole": (
        "short",
        "short circuit",
        "open_circuit",
        "open circuit",
        "단락",
        "단선",
        "패턴 연결",
        "패턴이 연결",
        "두 패턴이 연결",
        "두 회로 패턴 사이",
        "패턴 사이가 연결",
        "두 도전성 패턴 사이",
        "비정상적으로 이어",
        "브리지",
        "붙어 있",
        "패턴 끊김",
        "패턴이 끊",
        "회로가 끊",
    ),
}
LOCATION_LEAK_TERMS = (
    "상단",
    "하단",
    "좌측",
    "우측",
    "오른쪽",
    "중앙",
    "위쪽",
    "아래쪽",
    "왼쪽",
    "오른편",
    "top",
    "bottom",
    "left",
    "right",
    "center",
    "corner",
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
    class_conflict_count: int = 0
    class_conflict_detection_ids: tuple[int, ...] = ()
    location_leak_count: int = 0
    location_leak_detection_ids: tuple[int, ...] = ()
    language_warning_count: int = 0
    language_warning_detection_ids: tuple[int, ...] = ()
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
            corrected_response = apply_semantic_corrections(parsed_response, yolo_result)
            quality_info = _merge_corrected_summary_quality(
                quality_info,
                corrected_response,
                yolo_result,
            )
            formatted = format_parsed_vlm_response(corrected_response, yolo_result)
            return VlmParseResult(
                raw_response=raw_response,
                parse_success=True,
                parse_error="",
                fallback_used=False,
                parsed_response=corrected_response,
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
    class_conflict_ids: list[int] = []
    location_leak_ids: list[int] = []
    language_warning_ids: list[int] = []
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
        if has_class_conflict(vlm_detection.visual_feature, yolo_detection.class_name):
            class_conflict_ids.append(vlm_detection.detection_id)
        if has_location_leak(vlm_detection.visual_feature):
            location_leak_ids.append(vlm_detection.detection_id)
        if has_language_warning(vlm_detection.visual_feature):
            language_warning_ids.append(vlm_detection.detection_id)
    summary_contradiction = has_summary_contradiction(parsed_response, yolo_result)
    semantic_warning_count = (
        len(class_name_only_ids)
        + fallback_or_empty_count
        + len(class_conflict_ids)
        + len(location_leak_ids)
        + len(language_warning_ids)
        + int(summary_contradiction)
    )
    quality_status = "warning" if semantic_warning_count else "acceptable"
    return VlmQualityInfo(
        quality_status=quality_status,
        class_name_only_count=len(class_name_only_ids),
        class_name_only_detection_ids=tuple(class_name_only_ids),
        class_conflict_count=len(class_conflict_ids),
        class_conflict_detection_ids=tuple(class_conflict_ids),
        location_leak_count=len(location_leak_ids),
        location_leak_detection_ids=tuple(location_leak_ids),
        language_warning_count=len(language_warning_ids),
        language_warning_detection_ids=tuple(language_warning_ids),
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


def has_class_conflict(visual_feature: str, class_name: str) -> bool:
    """Return true when visual_feature mentions a different defect class."""
    normalized_class = _normalize_class_label(class_name)
    terms = CLASS_CONFLICT_TERMS.get(normalized_class, ())
    normalized_text = _normalize_search_text(visual_feature)
    return any(_contains_search_term(normalized_text, term) for term in terms)


def apply_semantic_corrections(
    parsed_response: ParsedVlmResponse,
    yolo_result: YoloResult,
) -> ParsedVlmResponse:
    """Replace only class-conflicting VLM explanations with conservative text."""
    corrected_detections: list[ParsedVlmDetection] = []
    changed = False
    for yolo_detection, vlm_detection in zip(
        yolo_result.detections,
        parsed_response.detections,
        strict=True,
    ):
        if has_class_conflict(vlm_detection.visual_feature, yolo_detection.class_name):
            corrected_detections.append(
                replace(
                    vlm_detection,
                    visual_feature=_safe_visual_feature_for_class(yolo_detection.class_name),
                    visibility="unclear",
                    review_required=True,
                )
            )
            changed = True
        else:
            corrected_detections.append(vlm_detection)

    if not changed:
        return parsed_response
    return replace(parsed_response, detections=corrected_detections)


def has_location_leak(visual_feature: str) -> bool:
    """Return true when visual_feature includes location words owned by YOLO metadata."""
    normalized_text = _normalize_search_text(visual_feature)
    return any(_contains_search_term(normalized_text, term) for term in LOCATION_LEAK_TERMS)


def has_language_warning(text: str) -> bool:
    """Return true for Chinese characters or English-heavy explanation text."""
    if re.search(r"[\u4e00-\u9fff]", text):
        return True
    normalized = _normalize_search_text(text)
    for class_name in ("short", "open_circuit", "open circuit", "missing_hole", "missing hole"):
        normalized = normalized.replace(class_name, " ")
    english_words = re.findall(r"\b[a-zA-Z]{2,}\b", normalized)
    if not english_words:
        return False
    has_hangul = re.search(r"[가-힣]", text) is not None
    return len(english_words) >= 3 or not has_hangul


def visual_feature_quality(visual_feature: str, class_name: str) -> str:
    """Classify one visual_feature quality with a small stable vocabulary."""
    if not visual_feature or not visual_feature.strip():
        return "empty"
    if visual_feature.strip() == ENGLISH_DEFAULT_VISUAL_FEATURE:
        return "fallback"
    if is_class_name_only_visual_feature(visual_feature, class_name):
        return "class_name_only"
    return "acceptable"


def has_summary_contradiction(
    parsed_response: ParsedVlmResponse,
    yolo_result: YoloResult,
) -> bool:
    """Detect summary claims that conflict with YOLO or parsed detection metadata."""
    normalized_summary = _normalize_search_text(parsed_response.summary)
    if has_summary_visibility_contradiction(parsed_response):
        return True
    if _summary_detection_count_conflicts(normalized_summary, len(parsed_response.detections)):
        return True
    if _summary_visibility_count_conflicts(parsed_response):
        return True
    if _summary_mentions_class_terms(normalized_summary):
        return True
    if _summary_mentions_confidence_or_bbox(normalized_summary):
        return True
    return any(
        has_class_conflict(parsed_response.summary, detection.class_name)
        for detection in yolo_result.detections
    )


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

    lines.extend(["", "종합 설명:", _format_deterministic_summary(parsed_response)])
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


def _format_deterministic_summary(parsed_response: ParsedVlmResponse) -> str:
    total_count = len(parsed_response.detections)
    clear_count = sum(
        1
        for detection in parsed_response.detections
        if detection.visibility == "clear" and not detection.review_required
    )
    review_count = sum(1 for detection in parsed_response.detections if detection.review_required)
    return (
        f"총 {total_count}개의 결함이 탐지되었으며, "
        f"{clear_count}개는 시각적 특징이 명확하고 "
        f"{review_count}개는 추가 확인이 필요합니다."
    )


def _safe_visual_feature_for_class(class_name: str) -> str:
    normalized_class = _normalize_class_label(class_name)
    return SAFE_VISUAL_FEATURE_BY_CLASS.get(normalized_class, DEFAULT_VISUAL_FEATURE)


def _merge_corrected_summary_quality(
    quality_info: VlmQualityInfo,
    corrected_response: ParsedVlmResponse,
    yolo_result: YoloResult,
) -> VlmQualityInfo:
    corrected_summary_contradiction = has_summary_contradiction(corrected_response, yolo_result)
    if not corrected_summary_contradiction or quality_info.summary_contradiction:
        return quality_info
    return replace(
        quality_info,
        quality_status="warning",
        summary_contradiction=True,
        semantic_warning_count=quality_info.semantic_warning_count + 1,
    )


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
    normalized = normalized.strip(" .:;,\n\t\r")
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = normalized.strip(" .:;,")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _normalize_search_text(value: str) -> str:
    normalized = value.lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _contains_search_term(normalized_text: str, term: str) -> bool:
    normalized_term = _normalize_search_text(term)
    if re.fullmatch(r"[a-z ]+", normalized_term):
        return re.search(rf"\b{re.escape(normalized_term)}\b", normalized_text) is not None
    return normalized_term in normalized_text


def _summary_detection_count_conflicts(summary: str, expected_count: int) -> bool:
    total_patterns = (
        r"총\s*(\d+)\s*개",
        r"전체\s*(\d+)\s*개",
        r"(\d+)\s*개의?\s*(?:결함|불량|탐지|detection|detections|defect|defects)",
        r"(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*(?:개의?\s*)?(?:결함|불량|탐지)",
    )
    for pattern in total_patterns:
        for match in re.finditer(pattern, summary):
            count = _parse_count_token(match.group(1))
            if count is not None and count != expected_count:
                return True
    return False


def _summary_visibility_count_conflicts(parsed_response: ParsedVlmResponse) -> bool:
    summary = parsed_response.summary.lower()
    clear_count = _extract_summary_count_before_terms(summary, ("명확", "clear"))
    unclear_count = _extract_summary_count_before_terms(summary, ("불명확", "추가 확인", "unclear", "review"))
    if clear_count is None and unclear_count is None:
        return False
    actual_clear = sum(1 for detection in parsed_response.detections if detection.visibility == "clear")
    actual_unclear = sum(1 for detection in parsed_response.detections if detection.visibility == "unclear")
    if clear_count is not None and clear_count != actual_clear:
        return True
    if unclear_count is not None and unclear_count != actual_unclear:
        return True
    if clear_count is not None and unclear_count is not None:
        return clear_count + unclear_count != len(parsed_response.detections)
    return False


def _summary_mentions_class_terms(summary: str) -> bool:
    terms = (
        "short",
        "open_circuit",
        "open circuit",
        "missing_hole",
        "missing hole",
        "단락",
        "단선",
        "누락된 홀",
    )
    return any(_normalize_search_text(term) in summary for term in terms)


def _summary_mentions_confidence_or_bbox(summary: str) -> bool:
    if any(term in summary for term in ("confidence", "신뢰도", "bbox", "bounding box", "바운딩", "좌표")):
        return True
    if re.search(r"\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", summary):
        return True
    return False


def _extract_summary_count_before_terms(summary: str, terms: tuple[str, ...]) -> int | None:
    for term in terms:
        term_index = summary.find(term)
        if term_index < 0:
            continue
        prefix = summary[:term_index]
        matches = list(
            re.finditer(
                r"(\d+|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*개?(?:는|은|가|이)?",
                prefix,
            )
        )
        if matches:
            return _parse_count_token(matches[-1].group(1))
    return None


def _parse_count_token(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    korean_counts = {
        "한": 1,
        "두": 2,
        "세": 3,
        "네": 4,
        "다섯": 5,
        "여섯": 6,
        "일곱": 7,
        "여덟": 8,
        "아홉": 9,
        "열": 10,
    }
    return korean_counts.get(value)


def _format_visibility(value: str) -> str:
    if value == "clear":
        return "명확함"
    if value == "unclear":
        return "불명확함"
    return value


def _format_review_required(value: bool) -> str:
    return "필요" if value else "불필요"
