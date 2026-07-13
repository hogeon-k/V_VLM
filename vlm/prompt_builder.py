from __future__ import annotations

from model.yolo_result import YoloResult


class PromptBuilder:
    """Build Korean prompts that keep YOLO as the source of truth."""

    def build_defect_prompt(self, yolo_result: YoloResult) -> str:
        detections = "\n".join(
            (
                f"{index}. class={detection.class_name}, confidence={detection.confidence:.4f}, "
                f"box=({detection.x1}, {detection.y1}, {detection.x2}, {detection.y2})"
            )
            for index, detection in enumerate(yolo_result.detections, start=1)
        )

        return f"""당신은 PCB 불량 검사 결과를 설명하는 보조자입니다.

YOLO 탐지 결과를 최우선 사실로 사용하세요. YOLO의 최종 불량 클래스, confidence, Bounding Box 좌표를 다시 분류하거나 변경하지 말고 설명만 하세요.
이미지에서 확인할 수 없는 원인, 제조 공정, 전기적 영향은 추측하지 마세요.
Bounding Box 좌표 자체만 나열하지 말고 상단, 하단, 좌측, 우측, 중앙 같은 자연스러운 위치 표현을 함께 사용하세요.
confidence가 낮은 탐지는 작업자 확인이 필요하다고 표시하세요.
여러 불량이 있으면 불량별로 구분하세요.
응답은 한국어로 작성하세요.

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
