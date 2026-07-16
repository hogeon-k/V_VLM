from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.console_encoding import configure_windows_console_encoding
from vlm.ollama_response import build_ollama_metadata
from vlm.response_parser import parse_vlm_response
from vlm.response_schema import VLM_RESPONSE_SCHEMA
from vlm.vlm_client import VlmClient

DEFAULT_IMAGE = Path(
    "data/result_images/montage/01_open_circuit_07_crop_montage_20260714_154328_506877.jpg"
)
DEFAULT_OPTIONS = {
    "num_ctx": 8192,
    "num_predict": 512,
    "temperature": 0.0,
    "top_p": 0.8,
    "top_k": 20,
    "repeat_penalty": 1.1,
    "seed": 42,
}
TEXT_PROMPT = "PCB가 무엇인지 한 문장으로 설명해줘."
IMAGE_PROMPT = "이미지에 보이는 내용을 한 문장으로 설명해줘."
SCHEMA_PROMPT = (
    "이미지를 보고 JSON만 반환하세요. "
    "final_judgment는 OK 또는 NG 중 하나를 사용하세요. "
    "detections 배열에는 detection_id=1인 항목 하나만 넣고, "
    "visual_feature, visibility, review_required를 포함하세요. "
    "summary를 포함하세요."
)


@dataclass(frozen=True)
class TestSpec:
    test_id: str
    test_name: str
    prompt: str
    image_count: int = 0
    include_options: bool = False
    format_value: str | dict[str, Any] | None = None


TEST_SPECS = (
    TestSpec("A", "text_minimal", TEXT_PROMPT),
    TestSpec("B", "text_options", TEXT_PROMPT, include_options=True),
    TestSpec("C", "image_no_format", IMAGE_PROMPT, image_count=1),
    TestSpec("D", "image_options", IMAGE_PROMPT, image_count=1, include_options=True),
    TestSpec(
        "E",
        "image_json_format",
        SCHEMA_PROMPT,
        image_count=1,
        include_options=True,
        format_value="json",
    ),
    TestSpec(
        "F",
        "image_schema",
        SCHEMA_PROMPT,
        image_count=1,
        include_options=True,
        format_value=VLM_RESPONSE_SCHEMA,
    ),
    TestSpec(
        "G",
        "two_images_schema",
        SCHEMA_PROMPT,
        image_count=2,
        include_options=True,
        format_value=VLM_RESPONSE_SCHEMA,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Ollama /api/chat payload conditions for qwen2.5vl vision calls."
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="First image path.")
    parser.add_argument("--second-image", default=None, help="Optional second image path for step G.")
    parser.add_argument("--model", default="qwen2.5vl:3b", help="Ollama model name.")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434", help="Ollama host URL.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--output-dir",
        default="data/result_images/ollama_diagnostics",
        help="Directory for per-step diagnostic JSON files.",
    )
    return parser.parse_args()


def build_payload(
    spec: TestSpec,
    *,
    model: str,
    encoded_images: list[str],
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "user",
        "content": spec.prompt,
    }
    if spec.image_count:
        message["images"] = encoded_images[: spec.image_count]

    payload: dict[str, Any] = {
        "model": model,
        "messages": [message],
        "stream": False,
    }
    if spec.include_options:
        payload["options"] = dict(DEFAULT_OPTIONS)
    if spec.format_value is not None:
        payload["format"] = spec.format_value
    return payload


def mask_base64_payload(payload: dict[str, Any]) -> dict[str, Any]:
    masked = json.loads(json.dumps(payload, ensure_ascii=False))
    for message in masked.get("messages", []):
        images = message.get("images")
        if isinstance(images, list):
            message["images"] = [
                f"<base64 omitted: {len(image)} chars>" if isinstance(image, str) else image
                for image in images
            ]
    return masked


def summarize_request(payload: dict[str, Any]) -> dict[str, Any]:
    message = payload["messages"][0]
    images = message.get("images", [])
    format_value = payload.get("format")
    return {
        "json_size_bytes": len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
        "base64_image_count": len(images) if isinstance(images, list) else 0,
        "base64_lengths": [len(image) for image in images] if isinstance(images, list) else [],
        "format": format_summary(format_value),
        "has_options": "options" in payload,
    }


def format_summary(value: object) -> str:
    if value is None:
        return "none"
    if value == "json":
        return "json"
    if isinstance(value, dict):
        return "JSON Schema object"
    return type(value).__name__


def parse_raw_response(raw_response: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return None, exc.msg
    if not isinstance(parsed, dict):
        return None, "response JSON root is not an object"
    return parsed, None


def evaluate_result(
    *,
    http_status: int | None,
    response_json: dict[str, Any] | None,
    require_json_content: bool,
    require_schema_content: bool,
) -> dict[str, Any]:
    if response_json is None:
        return {
            "success": False,
            "failure_reason": "response_json_parse_failed",
            "content_json_valid": None,
            "schema_valid": None,
            "schema_error": None,
        }

    message = response_json.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    done = response_json.get("done")
    content_text = content if isinstance(content, str) else ""
    content_json_valid: bool | None = None
    schema_valid: bool | None = None
    schema_error: str | None = None

    if require_json_content or require_schema_content:
        try:
            json.loads(content_text)
            content_json_valid = True
        except json.JSONDecodeError:
            content_json_valid = False

    if require_schema_content:
        try:
            parse_vlm_response(content_text, expected_detection_count=1)
            schema_valid = True
        except ValueError as exc:
            schema_valid = False
            schema_error = str(exc)

    if http_status != 200:
        failure_reason = f"http_status_{http_status}"
    elif not content_text.strip():
        failure_reason = "empty_message_content"
    elif done is not True:
        failure_reason = "done_not_true"
    else:
        failure_reason = None

    return {
        "success": failure_reason is None,
        "failure_reason": failure_reason,
        "content_json_valid": content_json_valid,
        "schema_valid": schema_valid,
        "schema_error": schema_error,
    }


def interpret_first_failure(results: list[dict[str, Any]]) -> str:
    success_by_id = {result["test_id"]: bool(result["success"]) for result in results}
    if not success_by_id.get("A", False):
        return "A 실패: Ollama API 또는 모델 기본 추론 문제"
    if not success_by_id.get("B", False):
        return "A 성공, B 실패: options 문제"
    if not success_by_id.get("C", False):
        return "B 성공, C 실패: vision 이미지 요청 또는 이미지 payload 문제"
    if not success_by_id.get("D", False):
        return "C 성공, D 실패: options와 vision 조합 문제"
    if not success_by_id.get("E", False):
        return "D 성공, E 실패: JSON mode와 vision 조합 문제"
    if not success_by_id.get("F", False):
        return "E 성공, F 실패: 전체 JSON Schema structured output 문제"
    if not success_by_id.get("G", False):
        return "F 성공, G 실패: 다중 이미지와 Schema 조합 문제"
    return "모두 성공: 기존 프로젝트 payload 또는 호출 경로와 진단 요청의 차이 재검토 필요"


def post_chat(
    *,
    host: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[int | None, str]:
    request = Request(
        url=f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(getattr(response, "status", 200)), response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except URLError as exc:
        return None, json.dumps({"error": f"URLError: {exc}"}, ensure_ascii=False)
    except TimeoutError as exc:
        return None, json.dumps({"error": f"TimeoutError: {exc}"}, ensure_ascii=False)


def save_result(output_dir: Path, record: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{record['test_id']}_{record['test_name']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def encode_images(image_paths: list[Path]) -> list[str]:
    return VlmClient()._encode_images_for_http([str(path) for path in image_paths])


def print_step_report(record: dict[str, Any]) -> None:
    response_json = record["response_json"] if isinstance(record["response_json"], dict) else {}
    message = response_json.get("message") if isinstance(response_json.get("message"), dict) else {}
    metadata = build_ollama_metadata(response_json, record["http_status"], "/api/chat", False)
    summary = record["request_summary"]
    print()
    print(f"테스트 단계: {record['test_id']}")
    print(f"테스트 이름: {record['test_name']}")
    print("endpoint: /api/chat")
    print(f"HTTP status: {record['http_status']}")
    print(f"요청 JSON 크기: {summary['json_size_bytes']}")
    print(f"base64 이미지 개수: {summary['base64_image_count']}")
    print(f"각 base64 문자열 길이: {summary['base64_lengths']}")
    print(f"format: {summary['format']}")
    print(f"options 포함 여부: {str(summary['has_options']).lower()}")
    print("응답 body 원문:")
    print(record["raw_response"])
    print(f"응답 JSON 파싱 성공 여부: {str(record['response_json_parse_success']).lower()}")
    print(f"model: {response_json.get('model')}")
    print(f"created_at: {response_json.get('created_at')}")
    print(f"done: {metadata.done}")
    print(f"done_reason: {metadata.done_reason}")
    print(f"error: {metadata.error}")
    print(f"message.role: {message.get('role')}")
    print(f"message.content 길이: {metadata.content_length}")
    print(f"prompt_eval_count: {metadata.prompt_eval_count}")
    print(f"eval_count: {metadata.eval_count}")
    print(f"total_duration: {metadata.total_duration}")
    print(f"content JSON 파싱 성공 여부: {record['content_json_valid']}")
    print(f"Schema 최소 검증 성공 여부: {record['schema_valid']}")
    print(f"성공/실패 판정: {'성공' if record['success'] else '실패'}")
    if record["failure_reason"]:
        print(f"실패 이유: {record['failure_reason']}")


def print_summary(results: list[dict[str, Any]]) -> None:
    print()
    print("ID | text/image | options | format | image_count | HTTP | done | content_length | result")
    for result in results:
        response_json = result["response_json"] if isinstance(result["response_json"], dict) else {}
        metadata = build_ollama_metadata(response_json, result["http_status"], "/api/chat", False)
        summary = result["request_summary"]
        text_or_image = "image" if summary["base64_image_count"] else "text"
        print(
            " | ".join(
                [
                    result["test_id"],
                    text_or_image,
                    "yes" if summary["has_options"] else "no",
                    summary["format"],
                    str(summary["base64_image_count"]),
                    str(result["http_status"]),
                    str(metadata.done),
                    str(metadata.content_length),
                    "success" if result["success"] else "fail",
                ]
            )
        )
    print()
    print(interpret_first_failure(results))


def run_diagnostics(args: argparse.Namespace) -> int:
    image_path = resolve_path(args.image)
    default_was_used = Path(args.image) == DEFAULT_IMAGE
    if not image_path.is_file():
        if default_was_used:
            print(
                "[ERROR] 기본 이미지가 존재하지 않습니다. "
                "--image로 진단에 사용할 이미지 경로를 지정해 주세요."
            )
            print(f"[ERROR] Missing default image: {image_path}")
        else:
            print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {image_path}")
        return 2

    second_image_path = resolve_path(args.second_image) if args.second_image else image_path
    if not second_image_path.is_file():
        print(f"[ERROR] 두 번째 이미지 파일을 찾을 수 없습니다: {second_image_path}")
        return 2
    if args.second_image is None:
        print("[INFO] --second-image가 없어 G 단계에서는 첫 번째 이미지를 한 번 더 사용합니다.")

    encoded_images = encode_images([image_path, second_image_path])
    output_dir = resolve_path(args.output_dir)
    results: list[dict[str, Any]] = []
    for spec in TEST_SPECS:
        payload = build_payload(spec, model=args.model, encoded_images=encoded_images)
        request_summary = summarize_request(payload)
        http_status, raw_response = post_chat(
            host=args.ollama_host,
            payload=payload,
            timeout=args.timeout,
        )
        response_json, parse_error = parse_raw_response(raw_response)
        evaluation = evaluate_result(
            http_status=http_status,
            response_json=response_json,
            require_json_content=spec.format_value == "json" or isinstance(spec.format_value, dict),
            require_schema_content=isinstance(spec.format_value, dict),
        )
        record = {
            "test_id": spec.test_id,
            "test_name": spec.test_name,
            "request": mask_base64_payload(payload),
            "request_summary": request_summary,
            "http_status": http_status,
            "raw_response": raw_response,
            "response_json": response_json or {},
            "response_json_parse_success": response_json is not None,
            "response_json_parse_error": parse_error,
            **evaluation,
        }
        save_result(output_dir, record)
        print_step_report(record)
        results.append(record)

    print_summary(results)
    print(f"[INFO] Diagnostic files: {output_dir.resolve()}")
    return 0


def main() -> int:
    configure_windows_console_encoding()
    return run_diagnostics(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
