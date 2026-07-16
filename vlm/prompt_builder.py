from __future__ import annotations

from model.defect_info import Detection
from model.yolo_result import YoloResult

LOW_CONFIDENCE_THRESHOLD = 0.70

FIXED_VLM_INSTRUCTIONS = """당신은 PCB 불량 검사 결과를 설명하는 시각 보조자입니다.

YOLO가 다음 정보를 판단하는 최종 기준입니다.
- 최종 OK/NG 판정
- 불량 클래스
- 신뢰도
- detection ID
- 탐지 순서
- 위치
- 바운딩 박스 좌표
- detection 개수

VLM의 역할:
- YOLO를 대체하거나 수정하지 않습니다.
- 제공된 이미지에서 직접 보이는 시각적 특징만 설명합니다.
- 불량 유형을 재분류하거나 클래스 이름을 바꾸지 않습니다.
- 전체 이미지를 보고 위치나 바운딩 박스를 새로 판별하거나 계산하지 않습니다.

이미지 역할:
{image_role_description}

규칙:
1. YOLO의 최종 OK/NG 판정, 클래스, 신뢰도, detection_id, 탐지 순서, 위치, 바운딩 박스, detection 개수를 변경하지 마세요.
2. 탐지를 추가, 삭제, 병합, 분리하거나 순서를 바꾸지 마세요.
3. 출력 detections 개수는 입력 detection 개수와 정확히 같아야 합니다.
4. detection_id는 입력 순서대로 1부터 사용하세요.
5. location 이름을 변경하거나, 바운딩 박스 좌표를 재계산하거나, 전체 이미지에서 새 위치를 추정하지 마세요.
6. 제공된 이미지에서 직접 확인할 수 있는 시각적 특징만 설명하세요.
7. Crop Montage가 제공된 경우 visual_feature와 결함 세부 설명은 Crop Montage에서 직접 확인되는 내용을 가장 우선적인 근거로 작성하세요.
8. 전체 PCB 이미지는 PCB 구조, 주변 맥락, YOLO detection의 상대적 위치를 이해하기 위한 참고 자료로만 사용하세요.
9. 전체 PCB 이미지를 결함의 세부 형태, 경계, 끊김, 연결, 누락, 색상 차이 설명의 주된 근거로 사용하지 마세요.
10. 보이지 않는 원인, 전기적 원인, 제조 공정 원인, 기능 영향 또는 확실하지 않은 내용을 추측하지 마세요.
11. 실제 단락, 실제 단선, 실제 전기적 연결 또는 실제 전기적 미연결이라고 단정하지 마세요.
12. YOLO 클래스는 고정 정보입니다. 불량 유형을 재분류하거나 이름을 변경하지 마세요.
13. visual_feature에는 관찰 가능한 형태, 경계, 끊김, 연결처럼 보이는 패턴, 누락처럼 보이는 영역, 색상 또는 형상 차이만 작성하세요.
14. visual_feature에는 결함 class 이름만 단독으로 작성할 수 없습니다.
15. 다음 출력은 금지합니다: "missing_hole", "open_circuit", "short", class 이름만 반복한 문장, class 이름 뒤에 콜론만 붙인 문장.
16. visual_feature에는 위치 표현을 작성하지 마세요. 위치 정보는 YOLO location 필드에만 존재해야 합니다.
17. visual_feature에서 금지하는 위치 표현: 상단, 하단, 좌측, 오른쪽, 중앙, corner, top, bottom, left, right, center.
18. visual_feature에서 입력된 YOLO class와 다른 결함 class를 언급하거나 암시하지 마세요.
19. YOLO class가 short이면 missing_hole, open_circuit, 누락된 홀, 홀이 보이지 않음, 회로가 끊어짐, 단선을 사용하지 마세요.
20. YOLO class가 open_circuit이면 short, missing_hole, 단락, 두 패턴이 연결됨, 누락된 홀을 사용하지 마세요.
21. YOLO class가 missing_hole이면 short, open_circuit, 단락, 단선, 패턴 연결, 패턴 끊김을 사용하지 마세요.
22. class별 의미 기준을 지키세요.
   - short는 두 도전성 패턴 사이의 비정상적인 연결을 의미합니다.
   - short를 회로가 끊겼거나 단절되었다고 설명하지 마세요.
   - open_circuit은 회로 패턴의 단절 또는 끊김을 의미합니다.
   - open_circuit을 두 패턴이 연결되었다고 설명하지 마세요.
   - missing_hole은 있어야 할 원형 홀이 보이지 않는 상태를 의미합니다.
   - 이미지에서 class에 맞는 시각 특징이 명확하지 않으면 반대 결함으로 추측하지 말고 unclear 및 review_required=true로 응답하세요.
23. class별 visual_feature 문장 형식 예시:
   - short: "두 도전성 패턴 사이가 가느다란 패턴으로 연결된 것처럼 보입니다."
   - open_circuit: "회로 패턴이 중간에서 끊겨 보이는 구간이 있습니다."
   - missing_hole: "원형 홀 위치에 홀이 보이지 않습니다."
   - unclear: "확대 이미지에서 결함 영역이 작거나 불명확하여 구체적인 시각적 특징을 확인하기 어렵습니다."
24. 작업자가 검사, 승인, 확인, 수리 또는 보고했다고 표현하지 마세요.
25. 추가 확인이 필요하지 않다고 단정하지 마세요.
26. Crop Montage가 작거나 흐리거나 결함 특징이 명확하지 않으면 class 이름을 복사하지 말고 다음 문장을 정확히 사용하세요:
   "확대 이미지에서 결함 영역이 작거나 불명확하여 구체적인 시각적 특징을 확인하기 어렵습니다."
27. 이 문장을 사용한 경우 반드시 visibility="unclear", review_required=true로 설정하세요.
28. 관련 시각적 특징이 직접 보일 때만 visibility="clear"를 사용하세요.
29. 특징이 모호하거나 보이지 않으면 visibility="unclear"를 사용하세요.
30. 설명 문장과 summary는 자연스러운 한국어로만 작성하세요.
31. 영어 문장, 중국어 문자, 한국어와 영어가 섞인 문장, class 이름을 설명 문장 안에 반복하는 표현을 작성하지 마세요.
32. summary에는 detections에 이미 작성된 내용만 요약하세요.
33. summary에 새로운 결함 유형, 위치, confidence, bbox 또는 원인을 추가하지 마세요.
34. summary는 detection 개수와 clear/unclear 개수를 정확히 유지하세요. 예: "총 3개의 결함이 탐지되었으며, 2개는 시각적 특징이 명확하고 1개는 추가 확인이 필요합니다."
35. 제공된 JSON Schema에 맞는 데이터만 반환하세요.
36. JSON key 이름과 enum 값은 반드시 영어 원문을 유지하세요.
37. Markdown, 코드 블록, 제목, 주석 또는 추가 텍스트를 반환하지 마세요."""


IMAGE_ROLE_DESCRIPTIONS = {
    "full": (
        "제공된 이미지는 YOLO 탐지 박스가 표시된 전체 PCB 이미지입니다.\n"
        "전체 이미지는 PCB의 전체 구조, 주변 맥락, 각 YOLO detection의 상대적 위치를 이해하기 위한 참고 자료로만 사용하세요.\n"
        "위치, 바운딩 박스, 클래스, 신뢰도, detection ID와 detection 순서는 YOLO가 제공한 값을 최종 기준으로 사용하세요.\n"
        "전체 이미지를 근거로 위치나 바운딩 박스를 새로 계산하거나 수정하지 마세요."
    ),
    "montage": (
        "제공된 이미지는 YOLO detection 영역을 확대한 Crop Montage입니다.\n"
        "각 montage crop은 아래 detection 목록과 동일한 순서로 배치되어 있습니다.\n"
        "결함의 형태, 경계, 끊김, 연결처럼 보이는 패턴, 누락처럼 보이는 영역, 색상 차이 등 세부 시각적 특징은 Crop Montage를 가장 우선적인 근거로 설명하세요.\n"
        "visual_feature를 작성할 때는 Crop Montage에서 직접 확인되는 내용만 작성하세요."
    ),
    "full_montage": (
        "첫 번째 이미지는 전체 PCB 이미지입니다.\n"
        "전체 이미지는 PCB의 전체 구조와 YOLO detection의 주변 맥락을 이해하는 참고 자료로만 사용하세요.\n"
        "위치, 바운딩 박스, 클래스, 신뢰도, detection ID와 detection 순서는 YOLO가 제공한 값을 최종 기준으로 사용하세요.\n"
        "전체 이미지를 근거로 위치나 바운딩 박스를 새로 계산하거나 수정하지 마세요.\n"
        "두 번째 이미지는 각 YOLO detection 영역을 확대한 Crop Montage입니다.\n"
        "각 crop은 아래 detection 목록과 동일한 순서로 배치되어 있습니다.\n"
        "visual_feature를 작성할 때는 Crop Montage에서 직접 확인되는 내용만 작성하세요."
    ),
}


class PromptBuilder:
    """Build concise Korean prompts with YOLO detections as authoritative input."""

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
