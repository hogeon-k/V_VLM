from __future__ import annotations

import json
import re
from typing import Any

from model.defect_info import Detection
from model.yolo_result import YoloResult
from vlm.prompt_builder import LOW_CONFIDENCE_THRESHOLD

DEFAULT_VISUAL_FEATURE = "crop 이미지에서 구체적인 시각적 특징을 명확히 확인하지 못했습니다."
DEFAULT_UNCERTAINTY = "이미지만으로 실제 결함 여부를 단정하기 어렵습니다."
DEFAULT_OPERATOR_CHECK = (
    "YOLO Bounding Box와 원본 이미지 및 crop 이미지를 함께 확인하여 "
    "실제 결함 여부를 최종 확인하세요."
)
_PARSE_FALLBACK_REASON = "VLM 응답을 구조화하지 못해 YOLO 결과 기준으로 정리했습니다."

_NO_CHECK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"no\s+check\s+required",
        r"confirmation\s+unnecessary",
        r"no\s+additional\s+action",
        r"no\s+further\s+action",
        r"not\s+required",
        r"필요\s*없음",
        r"확인\s*필요\s*없음",
        r"확인\s*불필요",
        r"추가\s*확인\s*불필요",
        r"추가\s*조치\s*없음",
    )
]

_UNCLEAR_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"image-only confirmation is difficult",
        r"difficult to confirm",
        r"unclear",
        r"not clearly visible",
        r"명확.*어렵",
        r"확인.*어렵",
        r"불명확",
    )
]


class VlmResponseParser:
    def parse_description(
        self,
        response_text: str,
        yolo_result: YoloResult | None = None,
    ) -> str:
        """Normalize provider responses while keeping YOLO output authoritative."""
        if yolo_result is None:
            return response_text.strip()
        return sanitize_vlm_explanation(response_text, yolo_result)


def sanitize_vlm_explanation(response_text: str, yolo_result: YoloResult) -> str:
    """Merge VLM observations into a YOLO-authoritative user explanation."""
    try:
        stripped = response_text.strip()
        parsed_json = _extract_json_object(stripped) if stripped else None
        if isinstance(parsed_json, dict):
            return _format_structured_response(
                vlm_data=parsed_json,
                yolo_result=yolo_result,
                parse_failed=False,
            )
        return _format_structured_response(
            vlm_data=None,
            yolo_result=yolo_result,
            parse_failed=True,
        )
    except Exception:
        return _format_structured_response(
            vlm_data=None,
            yolo_result=yolo_result,
            parse_failed=True,
        )


def _extract_json_object(response_text: str) -> dict[str, Any] | None:
    for candidate in _json_candidates(response_text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _json_candidates(response_text: str) -> list[str]:
    candidates = [response_text]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)```",
            response_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )

    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end > start:
        candidates.append(response_text[start : end + 1])

    unique_candidates = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _format_structured_response(
    vlm_data: dict[str, Any] | None,
    yolo_result: YoloResult,
    parse_failed: bool,
) -> str:
    vlm_detections_by_id = _detections_by_id(vlm_data)
    details = []
    operator_checks = []
    any_priority_recheck = parse_failed

    for index, detection in enumerate(yolo_result.detections, start=1):
        vlm_item = vlm_detections_by_id.get(index)
        merged = _merge_detection(index, detection, vlm_item, parse_failed)
        details.append(_format_detection_detail(index, detection, merged))
        if merged["operator_check"]:
            operator_checks.append(merged["operator_check"])
        any_priority_recheck = any_priority_recheck or bool(merged["priority_recheck"])

    overall_reason = _safe_text(_get_field(vlm_data, "overall_reason"))
    if not overall_reason:
        overall_reason = _build_default_overall_reason(yolo_result, parse_failed)

    final_operator_check = _safe_text(_get_field(vlm_data, "final_operator_check"))
    operator_checks.append(final_operator_check or DEFAULT_OPERATOR_CHECK)
    final_operator_check = _join_unique(operator_checks) or DEFAULT_OPERATOR_CHECK

    return _compose_response(
        detection_details=details,
        overall_reason=overall_reason,
        final_operator_check=final_operator_check,
        yolo_result=yolo_result,
        any_priority_recheck=any_priority_recheck,
    )


def _detections_by_id(vlm_data: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    detections = _get_field(vlm_data, "detections")
    if not isinstance(detections, list):
        return {}

    result = {}
    for item in detections:
        if not isinstance(item, dict):
            continue
        detection_id = _coerce_int(item.get("detection_id"))
        if detection_id is None:
            continue
        result[detection_id] = item
    return result


def _merge_detection(
    index: int,
    detection: Detection,
    vlm_item: dict[str, Any] | None,
    parse_failed: bool,
) -> dict[str, object]:
    visual_feature = _safe_text(_get_field(vlm_item, "visual_feature"))
    uncertainty = _safe_text(_get_field(vlm_item, "uncertainty"))
    operator_check = _sanitize_operator_check(
        _safe_text(_get_field(vlm_item, "operator_check"))
    )
    missing_vlm_result = vlm_item is None
    generic_observation = _looks_generic_visual_feature(detection.class_name, visual_feature)
    missing_observation = not visual_feature or generic_observation

    if missing_observation:
        visual_feature = DEFAULT_VISUAL_FEATURE
    if not uncertainty:
        uncertainty = DEFAULT_UNCERTAINTY if missing_observation else "특이 불확실성 언급 없음"
    if not operator_check:
        operator_check = DEFAULT_OPERATOR_CHECK

    priority_recheck = (
        detection.confidence < LOW_CONFIDENCE_THRESHOLD
        or missing_vlm_result
        or missing_observation
        or parse_failed
        or _as_bool(_get_field(vlm_item, "priority_recheck"))
        or _looks_unclear(visual_feature)
        or _looks_unclear(uncertainty)
    )

    return {
        "visual_feature": visual_feature,
        "uncertainty": uncertainty,
        "operator_check": operator_check,
        "priority_recheck": priority_recheck,
    }


def _format_detection_detail(
    index: int,
    detection: Detection,
    merged: dict[str, object],
) -> str:
    lines = [
        f"{index}. {detection.class_name}",
        f"   - 위치: {detection.location or '위치 미계산'}",
        f"   - 신뢰도: {detection.confidence:.4f}",
        f"   - Bounding Box: ({detection.x1}, {detection.y1}, {detection.x2}, {detection.y2})",
        f"   - 관찰된 시각적 특징: {merged['visual_feature']}",
        f"   - 불확실성: {merged['uncertainty']}",
    ]
    if merged["priority_recheck"]:
        lines.append("   - 우선 재검토 여부: 예")
    return "\n".join(lines)


def _compose_response(
    detection_details: list[str],
    overall_reason: str,
    final_operator_check: str,
    yolo_result: YoloResult,
    any_priority_recheck: bool,
) -> str:
    final_judgment = "NG" if yolo_result.is_ng else "OK"
    defect_classes = ", ".join(
        dict.fromkeys(detection.class_name for detection in yolo_result.detections)
    ) or "없음"
    details = "\n\n".join(detection_details) if detection_details else "탐지 없음"
    priority_line = "\n우선 재검토 대상이 포함되어 있습니다." if any_priority_recheck else ""
    return (
        f"최종 판정: {final_judgment}\n\n"
        f"탐지 요약:\n"
        f"- 클래스: {defect_classes}\n"
        f"- 탐지 수: {yolo_result.defect_count}\n\n"
        f"탐지 상세:\n\n{details}\n\n"
        f"종합 의견:\n{overall_reason}{priority_line}\n\n"
        f"최종 작업자 확인 사항:\n{final_operator_check}"
    )


def _build_default_overall_reason(yolo_result: YoloResult, parse_failed: bool) -> str:
    final_judgment = "NG" if yolo_result.is_ng else "OK"
    defect_classes = ", ".join(
        dict.fromkeys(detection.class_name for detection in yolo_result.detections)
    ) or "결함 없음"
    reason = f"YOLO가 {yolo_result.defect_count}개의 {defect_classes} 후보를 탐지하여 최종 판정은 {final_judgment}입니다."
    if parse_failed:
        reason = f"{reason} {_PARSE_FALLBACK_REASON}"
    return reason


def _get_field(data: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(data, dict):
        return None
    return data.get(key)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _sanitize_operator_check(value: str) -> str:
    if not value:
        return ""
    sanitized = value
    for pattern in _NO_CHECK_PATTERNS:
        sanitized = pattern.sub(DEFAULT_OPERATOR_CHECK, sanitized)
    return sanitized.strip()


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "예"}
    return False


def _looks_unclear(value: str) -> bool:
    return any(pattern.search(value) for pattern in _UNCLEAR_PATTERNS)


def _looks_generic_visual_feature(class_name: str, visual_feature: str) -> bool:
    lowered = visual_feature.lower()
    if class_name == "open_circuit":
        generic_parts = (
            "broken or missing circuit pattern",
            "discontinuity",
            "connection state",
            "differences from nearby normal traces",
        )
        return any(part in lowered for part in generic_parts)
    return False


def _join_unique(values: list[str]) -> str:
    unique_values = []
    for value in values:
        sanitized = _sanitize_operator_check(value)
        if sanitized and sanitized not in unique_values:
            unique_values.append(sanitized)
    return "\n".join(f"- {value}" for value in unique_values)
