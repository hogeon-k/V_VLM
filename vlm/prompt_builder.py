from __future__ import annotations

from model.yolo_result import YoloResult


class PromptBuilder:
    """Build Korean prompts that keep YOLO and Python location data authoritative."""

    def build_defect_prompt(self, yolo_result: YoloResult) -> str:
        detections = "\n\n".join(
            (
                f"탐지 {index}\n"
                f"- 클래스: {detection.class_name}\n"
                f"- 신뢰도: {detection.confidence:.4f}\n"
                f"- 위치: {detection.location or '위치 미계산'}\n"
                f"- Bounding Box: ({detection.x1}, {detection.y1}, {detection.x2}, {detection.y2})"
            )
            for index, detection in enumerate(yolo_result.detections, start=1)
        )

        return f"""당신은 PCB 불량 검사 결과를 설명하는 보조자입니다.

입력 이미지 설명:
- 첫 번째 이미지는 PCB 전체 위치와 YOLO Bounding Box를 확인하기 위한 전체 결과 이미지입니다.
- 두 번째 이미지는 각 탐지 영역의 원본 해상도 crop을 하나로 합친 crop montage입니다.
- crop montage의 Detection 1, Detection 2, Detection 3은 아래 탐지 정보 순서와 동일합니다.

YOLO 탐지 결과는 불량 후보 정보입니다.
- 아래의 클래스, 신뢰도, 위치, Bounding Box는 Python과 YOLO가 제공한 값이므로 변경하지 마세요.
- 위치는 Python 코드에서 계산한 값이므로 다시 추론하지 마세요.
- 좌표를 이용해 새로운 위치를 계산하지 마세요.
- 전체 이미지의 위치는 맥락 확인에 사용하세요.
- crop montage는 각 탐지 영역의 실제 회로 패턴 단절 여부를 확인하는 데 사용하세요.
- crop에서 결함 형태가 명확하지 않으면 확정적으로 판정하지 마세요.
- YOLO 탐지 후보와 crop에서 시각적으로 확인한 내용을 구분해서 설명하세요.
- 모든 Detection을 순서대로 확인하세요.
- 한 탐지의 시각 특징을 다른 탐지에 잘못 적용하지 마세요.
- "녹색 PCB가 보입니다" 같은 일반적인 설명은 피하세요.
- 시각적으로 확인하기 어려운 내용은 "이미지만으로 확인 어려움"이라고 작성하세요.
- 축소 이미지 또는 crop에서도 세부 단절 형태 확인이 어려우면 "축소 이미지에서는 세부 단절 형태 확인이 어려움"이라고 작성하세요.
- 응답은 한국어로 작성하세요.
- 응답은 6개 항목 형식을 유지하되 각 항목은 짧게 작성하세요.

YOLO 탐지 목록:
{detections}

아래 형식을 지키세요.
1. 최종 판정: NG
2. 탐지된 불량
3. 불량 위치
4. 시각적 특징
5. 판정 근거
6. 작업자 확인 사항
"""
