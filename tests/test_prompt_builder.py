from __future__ import annotations

from model.defect_info import Detection
from model.yolo_result import YoloResult
from vlm.prompt_builder import PromptBuilder


def test_prompt_builder_includes_calculated_location_and_rules() -> None:
    result = YoloResult(
        image_path="sample.jpg",
        detections=[
            Detection(0, "open_circuit", 0.784, 2711, 946, 2739, 979, location="중단 오른쪽"),
            Detection(0, "open_circuit", 0.7583, 2547, 540, 2595, 584, location="상단 오른쪽"),
        ],
    )

    prompt = PromptBuilder().build_defect_prompt(result)

    assert "- 클래스: open_circuit" in prompt
    assert "- 신뢰도: 0.7840" in prompt
    assert "- 위치: 중단 오른쪽" in prompt
    assert "- Bounding Box: (2711, 946, 2739, 979)" in prompt
    assert "탐지 1" in prompt
    assert "탐지 2" in prompt
    assert "위치는 Python 코드에서 계산한 값이므로 다시 추론하지 마세요." in prompt
    assert "좌표를 이용해 새로운 위치를 계산하지 마세요." in prompt
    assert '"녹색 PCB가 보입니다" 같은 일반적인 설명은 피하세요.' in prompt
    assert "이미지만으로 확인 어려움" in prompt
    assert "축소 이미지에서는 세부 단절 형태 확인이 어려움" in prompt
