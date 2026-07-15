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
    assert "- detection ID" in prompt
    assert "- detection 개수" in prompt
    assert "전체 이미지를 보고 위치나 바운딩 박스를 새로 판별하거나 계산하지 않습니다." in prompt
    assert "첫 번째 이미지는 전체 PCB 이미지입니다." in prompt
    assert "전체 이미지는 PCB의 전체 구조와 YOLO detection의 주변 맥락을 이해하는 참고 자료로만 사용하세요." in prompt
    assert "전체 이미지를 근거로 위치나 바운딩 박스를 새로 계산하거나 수정하지 마세요." in prompt
    assert "두 번째 이미지는 각 YOLO detection 영역을 확대한 Crop Montage입니다." in prompt
    assert "각 crop은 아래 detection 목록과 동일한 순서로 배치되어 있습니다." in prompt
    assert "visual_feature를 작성할 때는 Crop Montage에서 직접 확인되는 내용만 작성하세요." in prompt
    assert "location 이름을 변경하거나, 바운딩 박스 좌표를 재계산하거나, 전체 이미지에서 새 위치를 추정하지 마세요." in prompt
    assert "YOLO 클래스는 고정 정보입니다." in prompt
    assert "visual_feature에는 결함 class 이름만 단독으로 작성할 수 없습니다." in prompt
    assert '"missing_hole", "open_circuit", "short"' in prompt
    assert "visual_feature에는 위치 표현을 작성하지 마세요." in prompt
    assert "YOLO class가 short이면 missing_hole, open_circuit, 누락된 홀, 홀이 보이지 않음, 회로가 끊어짐, 단선을 사용하지 마세요." in prompt
    assert "회로 패턴이 중간에서 끊겨 보이는 구간이 있습니다." in prompt
    assert "보이지 않는 원인, 전기적 원인, 제조 공정 원인, 기능 영향 또는 확실하지 않은 내용을 추측하지 마세요." in prompt
    assert "실제 단락, 실제 단선, 실제 전기적 연결 또는 실제 전기적 미연결이라고 단정하지 마세요." in prompt
    assert "확대 이미지에서 결함 영역이 작거나 불명확하여 구체적인 시각적 특징을 확인하기 어렵습니다." in prompt
    assert 'visibility="unclear", review_required=true' in prompt
    assert "설명 문장과 summary는 자연스러운 한국어로만 작성하세요." in prompt
    assert "summary에는 detections에 이미 작성된 내용만 요약하세요." in prompt
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
    assert "Crop Montage" in montage_prompt
    assert "전체 이미지를 근거로 위치나 바운딩 박스를 새로 계산하거나 수정하지 마세요." in full_prompt
    assert "visual_feature를 작성할 때는 Crop Montage에서 직접 확인되는 내용만 작성하세요." in montage_prompt
