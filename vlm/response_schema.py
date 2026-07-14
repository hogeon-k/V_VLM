from __future__ import annotations

VLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "final_judgment": {
            "type": "string",
            "enum": ["OK", "NG"],
        },
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "detection_id": {
                        "type": "integer",
                    },
                    "visual_feature": {
                        "type": "string",
                        "description": (
                            "이미지에서 관찰 가능한 시각적 형태 또는 패턴. YOLO 클래스명만 쓰지 말 것."
                        ),
                        "minLength": 1,
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["clear", "unclear"],
                    },
                    "review_required": {
                        "type": "boolean",
                    },
                },
                "required": [
                    "detection_id",
                    "visual_feature",
                    "visibility",
                    "review_required",
                ],
                "additionalProperties": False,
            },
        },
        "summary": {
            "type": "string",
        },
    },
    "required": [
        "final_judgment",
        "detections",
        "summary",
    ],
    "additionalProperties": False,
}
