from __future__ import annotations

import json

import scripts.diagnose_ollama_payload as diag
from vlm.response_schema import VLM_RESPONSE_SCHEMA


ENCODED_IMAGES = ["abc123", "def456789"]


def payload_for(test_id: str) -> dict[str, object]:
    spec = next(item for item in diag.TEST_SPECS if item.test_id == test_id)
    return diag.build_payload(spec, model="qwen2.5vl:3b", encoded_images=ENCODED_IMAGES)


def test_builds_payload_a_text_minimal() -> None:
    payload = payload_for("A")

    assert payload == {
        "model": "qwen2.5vl:3b",
        "messages": [{"role": "user", "content": diag.TEXT_PROMPT}],
        "stream": False,
    }


def test_builds_payload_b_text_options() -> None:
    payload = payload_for("B")

    assert payload["options"] == diag.DEFAULT_OPTIONS
    assert "images" not in payload["messages"][0]
    assert "format" not in payload


def test_image_payloads_place_images_inside_user_message() -> None:
    payload = payload_for("C")

    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["images"] == ["abc123"]
    assert "format" not in payload
    assert "options" not in payload


def test_payload_d_adds_options_to_vision_request() -> None:
    payload = payload_for("D")

    assert payload["messages"][0]["images"] == ["abc123"]
    assert payload["options"] == diag.DEFAULT_OPTIONS
    assert "format" not in payload


def test_payload_e_uses_json_format() -> None:
    payload = payload_for("E")

    assert payload["messages"][0]["images"] == ["abc123"]
    assert payload["format"] == "json"
    assert payload["options"] == diag.DEFAULT_OPTIONS


def test_payload_f_uses_full_schema() -> None:
    payload = payload_for("F")

    assert payload["messages"][0]["images"] == ["abc123"]
    assert payload["format"] == VLM_RESPONSE_SCHEMA


def test_payload_g_uses_two_images_and_full_schema() -> None:
    payload = payload_for("G")

    assert payload["messages"][0]["images"] == ENCODED_IMAGES
    assert payload["format"] == VLM_RESPONSE_SCHEMA


def test_mask_base64_payload_replaces_image_values_with_lengths() -> None:
    masked = diag.mask_base64_payload(payload_for("G"))

    assert masked["messages"][0]["images"] == [
        "<base64 omitted: 6 chars>",
        "<base64 omitted: 9 chars>",
    ]


def test_summarize_request_reports_format_and_image_lengths() -> None:
    summary = diag.summarize_request(payload_for("F"))

    assert summary["base64_image_count"] == 1
    assert summary["base64_lengths"] == [6]
    assert summary["format"] == "JSON Schema object"
    assert summary["has_options"] is True
    assert summary["json_size_bytes"] > 0


def test_evaluate_result_success_requires_http_content_and_done_true() -> None:
    response_json = {"message": {"role": "assistant", "content": "ok"}, "done": True}

    result = diag.evaluate_result(
        http_status=200,
        response_json=response_json,
        require_json_content=False,
        require_schema_content=False,
    )

    assert result["success"] is True
    assert result["failure_reason"] is None


def test_evaluate_result_rejects_done_false() -> None:
    response_json = {"message": {"role": "assistant", "content": "ok"}, "done": False}

    result = diag.evaluate_result(
        http_status=200,
        response_json=response_json,
        require_json_content=False,
        require_schema_content=False,
    )

    assert result["success"] is False
    assert result["failure_reason"] == "done_not_true"


def test_evaluate_result_records_json_content_validity() -> None:
    response_json = {"message": {"role": "assistant", "content": '{"summary":"ok"}'}, "done": True}

    result = diag.evaluate_result(
        http_status=200,
        response_json=response_json,
        require_json_content=True,
        require_schema_content=False,
    )

    assert result["success"] is True
    assert result["content_json_valid"] is True


def test_evaluate_result_records_schema_validation() -> None:
    content = json.dumps(
        {
            "final_judgment": "NG",
            "detections": [
                {
                    "detection_id": 1,
                    "visual_feature": "A visible gap is present.",
                    "visibility": "clear",
                    "review_required": False,
                }
            ],
            "summary": "One defect was described.",
        }
    )
    response_json = {"message": {"role": "assistant", "content": content}, "done": True}

    result = diag.evaluate_result(
        http_status=200,
        response_json=response_json,
        require_json_content=True,
        require_schema_content=True,
    )

    assert result["success"] is True
    assert result["content_json_valid"] is True
    assert result["schema_valid"] is True
    assert result["schema_error"] is None


def test_parse_raw_response_handles_invalid_json() -> None:
    response_json, error = diag.parse_raw_response("{bad json")

    assert response_json is None
    assert error is not None


def test_interpret_first_failure_messages() -> None:
    results = [{"test_id": item.test_id, "success": True} for item in diag.TEST_SPECS]
    results[5]["success"] = False
    results[6]["success"] = False

    assert diag.interpret_first_failure(results) == (
        "E 성공, F 실패: 전체 JSON Schema structured output 문제"
    )

    assert diag.interpret_first_failure([{"test_id": item.test_id, "success": True} for item in diag.TEST_SPECS]) == (
        "모두 성공: 기존 프로젝트 payload 또는 호출 경로와 진단 요청의 차이 재검토 필요"
    )
