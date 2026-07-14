from __future__ import annotations

from model.defect_info import Detection
from model.yolo_result import YoloResult
from vlm.prompt_builder import FIXED_VLM_INSTRUCTIONS, PromptBuilder


def test_prompt_builder_includes_korean_rules_and_detection_metadata() -> None:
    result = YoloResult(
        image_path="sample.jpg",
        detections=[
            Detection(0, "open_circuit", 0.784, 2711, 946, 2739, 979, location="middle right"),
            Detection(0, "open_circuit", 0.6696, 2193, 294, 2230, 326, location="upper right"),
        ],
    )

    prompt = PromptBuilder().build_defect_prompt(result)

    assert prompt.startswith("당신은 PCB 불량 검사 결과를 설명하는 시각 보조자입니다.")
    assert "{image_role_description}" in FIXED_VLM_INSTRUCTIONS
    assert "YOLO가 다음 정보를 판단하는 최종 기준입니다." in prompt
    assert "YOLO 클래스는 고정 정보입니다." in prompt
    assert "visual_feature를 YOLO 클래스명만으로 작성하지 마세요." in prompt
    assert "전도성 패턴 중간에 끊어진 구간이 보입니다." in prompt
    assert "제공된 JSON Schema에 맞는 데이터만 반환하세요." in prompt
    assert "JSON key 이름과 enum 값은 반드시 영어 원문을 유지하세요." in prompt
    assert "최종 판정: NG" in prompt
    assert "탐지 개수: 2" in prompt
    assert "탐지 1" in prompt
    assert "클래스: open_circuit" in prompt
    assert "신뢰도: 0.7840" in prompt
    assert "위치: middle right" in prompt
    assert "바운딩 박스: (2711, 946, 2739, 979)" in prompt
    assert "탐지 2" in prompt
    assert "위치: upper right" in prompt


def test_prompt_builder_uses_location_unavailable_when_missing() -> None:
    result = YoloResult(
        image_path="sample.jpg",
        detections=[Detection(0, "open_circuit", 0.6696, 1, 2, 3, 4)],
    )

    prompt = PromptBuilder().build_defect_prompt(result)

    assert "위치: 위치 정보 없음" in prompt


def test_prompt_builder_describes_image_role_by_mode() -> None:
    result = YoloResult(
        image_path="sample.jpg",
        detections=[Detection(0, "short", 0.9, 1, 2, 3, 4)],
    )

    full_prompt = PromptBuilder().build_defect_prompt(result, image_mode="full")
    montage_prompt = PromptBuilder().build_defect_prompt(result, image_mode="montage")

    assert "전체 PCB 이미지" in full_prompt
    assert "crop montage" in montage_prompt
