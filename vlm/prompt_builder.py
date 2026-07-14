from __future__ import annotations

from model.defect_info import Detection
from model.yolo_result import YoloResult

LOW_CONFIDENCE_THRESHOLD = 0.70

FIXED_VLM_INSTRUCTIONS = """당신은 PCB 불량 검사 결과를 설명하는 시각 보조자입니다.

YOLO가 다음 정보를 판단하는 최종 기준입니다.
- 최종 OK/NG 판정
- 불량 클래스
- 신뢰도
- 탐지 순서
- 위치
- 바운딩 박스 좌표

VLM의 역할:
- YOLO를 대체하거나 수정하지 않습니다.
- 제공된 이미지에서 직접 보이는 시각적 특징만 설명합니다.
- 불량 유형을 재분류하거나 클래스 이름을 바꾸지 않습니다.

이미지 역할:
{image_role_description}

규칙:
1. YOLO 클래스, 신뢰도, 위치, 바운딩 박스, detection_id를 변경하지 마세요.
2. 탐지를 추가, 삭제, 병합, 분리하거나 순서를 바꾸지 마세요.
3. 출력 detections 개수는 입력 detection 개수와 정확히 같아야 합니다.
4. detection_id는 입력 순서대로 1부터 사용하세요.
5. 제공된 이미지에서 직접 확인할 수 있는 시각적 특징만 설명하세요.
6. YOLO 클래스는 고정 정보입니다. 불량 유형을 재분류하거나 이름을 변경하지 마세요.
7. visual_feature에는 관찰 가능한 형태나 패턴만 작성하세요.
8. visual_feature를 YOLO 클래스명만으로 작성하지 마세요.
9. "short", "open_circuit", "missing_hole", "<class> defect"만 작성하지 마세요.
10. 좋은 visual_feature 예시:
   - short: "인접한 구리 패턴 사이에 가느다란 전기적 연결부가 보입니다."
   - open_circuit: "전도성 패턴 중간에 끊어진 구간이 보입니다."
   - missing_hole: "원형 패드 영역에서 예상되는 천공 구멍이 보이지 않습니다."
11. 작업자가 검사, 승인, 확인, 수리 또는 보고했다고 표현하지 마세요.
12. 추가 확인이 필요하지 않다고 단정하지 마세요.
13. 명확한 시각적 특징을 확인할 수 없으면 다음 문장을 사용하세요:
   "명확한 시각적 특징을 확인하지 못했습니다."
14. 관련 시각적 특징이 직접 보일 때만 visibility="clear"를 사용하세요.
15. 특징이 모호하거나 보이지 않으면 visibility="unclear"를 사용하세요.
16. visibility가 "unclear"이면 review_required=true로 설정하세요.
17. 제공된 JSON Schema에 맞는 데이터만 반환하세요.
18. JSON key 이름과 enum 값은 반드시 영어 원문을 유지하세요.
19. Markdown, 코드 블록, 제목, 주석 또는 추가 텍스트를 반환하지 마세요."""


IMAGE_ROLE_DESCRIPTIONS = {
    "full": "제공된 이미지는 YOLO 탐지 박스가 표시된 전체 PCB 이미지입니다.",
    "montage": (
        "제공된 이미지는 YOLO 탐지 영역을 확대한 crop montage입니다.\n"
        "montage crop 순서는 아래 detection 목록 순서와 같습니다."
    ),
    "full_montage": (
        "첫 번째 이미지는 YOLO 탐지 박스가 표시된 전체 PCB 이미지입니다.\n"
        "두 번째 이미지는 탐지 영역을 확대한 crop montage입니다.\n"
        "montage crop 순서는 아래 detection 목록 순서와 같습니다."
    ),
}


class PromptBuilder:
    """Build concise English prompts with YOLO detections as authoritative input."""

    def build_defect_prompt(self, yolo_result: YoloResult, image_mode: str = "full_montage") -> str:
        final_judgment = "NG" if yolo_result.is_ng else "OK"
        detection_blocks = "\n\n".join(
            self._format_detection(index, detection)
            for index, detection in enumerate(yolo_result.detections, start=1)
        )
        instructions = FIXED_VLM_INSTRUCTIONS.format(
            image_role_description=IMAGE_ROLE_DESCRIPTIONS.get(
                image_mode,
                "제공된 이미지를 기준으로 아래 detection 목록 순서에 맞춰 각 영역을 설명하세요.",
            )
        )
        return (
            f"{instructions}\n\n"
            "YOLO 탐지 결과:\n\n"
            f"최종 판정: {final_judgment}\n"
            f"탐지 개수: {yolo_result.defect_count}\n\n"
            f"{detection_blocks}"
        ).strip()

    def _format_detection(self, index: int, detection: Detection) -> str:
        return (
            f"탐지 {index}\n"
            f"클래스: {detection.class_name}\n"
            f"신뢰도: {detection.confidence:.4f}\n"
            f"위치: {detection.location or '위치 정보 없음'}\n"
            f"바운딩 박스: ({detection.x1}, {detection.y1}, {detection.x2}, {detection.y2})"
        )
