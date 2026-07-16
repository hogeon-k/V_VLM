from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.yolo_result import YoloResult
from scripts.console_encoding import configure_windows_console_encoding
from scripts.test_yolo_vlm import build_vlm_service, build_yolo_service, positive_int
from vlm.ollama_response import OllamaResponseMetadata
from vlm.response_parser import format_yolo_fallback_response

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GROUND_TRUTH_CATEGORIES = {"open_circuit", "short", "missing_hole", "normal"}
SKIPPED_VLM_MESSAGE = "VLM analysis skipped because no defect was detected."
CSV_COLUMNS = [
    "category",
    "image_name",
    "image_path",
    "ground_truth_class",
    "yolo_judgment",
    "yolo_detection_count",
    "yolo_classes",
    "yolo_confidences",
    "annotated_image_path",
    "crop_montage_path",
    "vlm_model",
    "vlm_response",
    "vlm_raw_response",
    "vlm_parse_success",
    "vlm_parse_error",
    "vlm_fallback_used",
    "vlm_temperature",
    "vlm_top_p",
    "vlm_top_k",
    "vlm_repeat_penalty",
    "vlm_seed",
    "vlm_image_mode",
    "image_preparation_time_seconds",
    "vlm_inference_time_seconds",
    "total_processing_time_seconds",
    "status",
    "image_status",
    "error_message",
    "pipeline_status",
    "yolo_status",
    "vlm_status",
    "parse_status",
    "fallback_used",
    "retry_count",
    "failure_reason",
    "vlm_error_type",
    "vlm_error_message",
    "pipeline_success",
    "yolo_success",
    "vlm_attempted",
    "vlm_success",
    "result_saved",
    "http_status",
    "ollama_endpoint",
    "ollama_stream",
    "ollama_error",
    "ollama_done",
    "ollama_done_reason",
    "ollama_content_length",
    "ollama_prompt_eval_count",
    "ollama_eval_count",
    "ollama_total_duration",
    "ollama_load_duration",
    "ollama_prompt_eval_duration",
    "ollama_eval_duration",
    "vlm_image_count",
    "crop_count",
    "vlm_full_image_size_limit",
    "vlm_montage_size_limit",
    "vlm_full_image_width",
    "vlm_full_image_height",
    "montage_width",
    "montage_height",
    "quality_status",
    "class_name_only_count",
    "class_conflict_count",
    "location_leak_count",
    "language_warning_count",
    "summary_contradiction",
    "semantic_warning_count",
    "class_name_only_detection_ids",
    "class_conflict_detection_ids",
    "location_leak_detection_ids",
    "language_warning_detection_ids",
    "exception_type",
    "exception_message",
]


def parse_args() -> argparse.Namespace:
    """Parse batch YOLO + VLM test options."""
    parser = argparse.ArgumentParser(
        description="Run YOLO and optional Ollama VLM over a directory of test images."
    )
    parser.add_argument("--input-dir", default="data/vlm_test_images", help="Root directory containing test images.")
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU threshold.")
    parser.add_argument("--device", default="0", help="YOLO device, for example 0 or cpu.")
    parser.add_argument("--vlm-model", default="qwen2.5vl:3b", help="Ollama VLM model name.")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434", help="Ollama host URL.")
    parser.add_argument("--vlm-num-ctx", type=int, default=8192, help="Ollama VLM context window.")
    parser.add_argument("--vlm-num-predict", type=int, default=256, help="Ollama VLM max generated tokens.")
    parser.add_argument("--vlm-temperature", type=float, default=0.0, help="Ollama generation temperature.")
    parser.add_argument("--vlm-top-p", type=float, default=0.8, help="Ollama nucleus sampling probability.")
    parser.add_argument("--vlm-top-k", type=int, default=20, help="Ollama top-k sampling value.")
    parser.add_argument("--vlm-repeat-penalty", type=float, default=1.1, help="Ollama repetition penalty.")
    parser.add_argument("--vlm-seed", type=int, default=42, help="Ollama random seed for repeatable generation.")
    parser.add_argument(
        "--vlm-debug-response",
        action="store_true",
        help="Print safe Ollama response structure diagnostics.",
    )
    parser.add_argument("--vlm-image-size", type=positive_int, default=960, help="Max VLM input image side length.")
    parser.add_argument(
        "--vlm-full-image-size",
        type=positive_int,
        default=None,
        help="Max VLM full image side length. Overrides --vlm-image-size when set.",
    )
    parser.add_argument("--vlm-image-quality", type=int, default=90, help="VLM JPEG input quality.")
    parser.add_argument(
        "--vlm-image-mode",
        choices=("full", "montage", "full_montage"),
        default="full_montage",
        help="VLM image payload mode.",
    )
    parser.add_argument("--vlm-crop-montage-size", type=positive_int, default=960, help="Max VLM crop montage side length.")
    parser.add_argument(
        "--vlm-montage-size",
        type=positive_int,
        default=None,
        help="Max VLM crop montage side length. Overrides --vlm-crop-montage-size when set.",
    )
    parser.add_argument("--vlm-crop-padding", type=int, default=192, help="Detection crop padding target in pixels.")
    parser.add_argument("--vlm-crop-min-size", type=int, default=256, help="Minimum detection crop side length.")
    parser.add_argument("--vlm-crop-max-size", type=int, default=512, help="Maximum detection crop side length.")
    parser.add_argument(
        "--save-crop-montage",
        action="store_true",
        help="Save generated detection crop montage images for debugging.",
    )
    parser.add_argument(
        "--crop-montage-output-dir",
        default="data/result_images/montage",
        help="Directory used when --save-crop-montage is enabled.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/result_images/vlm_batch_results",
        help="Directory for the batch result CSV.",
    )
    parser.add_argument("--vlm-max-retries", type=int, default=2, help="Maximum VLM retry count per image.")
    parser.add_argument("--vlm-retry-delay", type=float, default=0.5, help="Delay between VLM retries in seconds.")
    parser.add_argument("--vlm-timeout", type=float, default=120.0, help="Ollama HTTP timeout in seconds.")
    parser.add_argument(
        "--vlm-circuit-breaker-threshold",
        type=int,
        default=3,
        help="Consecutive zero-value VLM fallbacks before skipping later VLM calls.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue processing later images after an image-level failure.",
    )
    parser.add_argument(
        "--save-raw-response-on-failure",
        action="store_true",
        default=True,
        help="Save raw VLM response text when an image uses fallback or fails.",
    )
    return parser.parse_args()


def discover_images(input_dir: Path) -> list[Path]:
    """Return supported image files recursively in stable path order."""
    if not input_dir.exists():
        return []
    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(input_dir).as_posix().lower(),
    )


def ground_truth_for_category(category: str) -> str:
    """Map folder category to a ground truth class when it is unambiguous."""
    return category if category in GROUND_TRUTH_CATEGORIES else ""


def build_csv_path(output_dir: Path) -> Path:
    """Create the output directory and return a timestamped CSV path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"vlm_batch_results_{timestamp}.csv"


def build_batch_paths(output_dir: Path) -> dict[str, Path]:
    """Create the batch output folders used for immediate per-image results."""
    paths = {
        "root": output_dir,
        "results": output_dir / "results",
        "result_images": output_dir / "result_images",
        "montage": output_dir / "montage",
        "raw_responses": output_dir / "raw_responses",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def result_to_row(
    *,
    image_path: Path,
    input_dir: Path,
    yolo_result: YoloResult | None,
    vlm_model: str,
    vlm_response: str,
    crop_montage_path: Path | None = None,
    image_preparation_seconds: float | None = None,
    vlm_inference_seconds: float | None = None,
    total_processing_seconds: float = 0.0,
    status: str = "success",
    image_status: str = "completed",
    vlm_raw_response: str = "",
    vlm_parse_success: bool = False,
    vlm_parse_error: str = "",
    vlm_fallback_used: bool = False,
    vlm_temperature: float = 0.0,
    vlm_top_p: float = 0.8,
    vlm_top_k: int = 20,
    vlm_repeat_penalty: float = 1.1,
    vlm_seed: int = 42,
    vlm_image_mode: str = "",
    error_message: str = "",
    pipeline_status: str | None = None,
    yolo_status: str = "success",
    vlm_status: str = "not_run",
    parse_status: str = "not_attempted",
    retry_count: int = 0,
    failure_reason: str = "",
    vlm_error_type: str = "",
    vlm_error_message: str = "",
    result_saved: bool = True,
    ollama_metadata: OllamaResponseMetadata | None = None,
    vlm_image_count: int | None = None,
    crop_count: int | None = None,
    full_image_size_limit: int | None = None,
    montage_size_limit: int | None = None,
    full_image_size: tuple[int, int] | None = None,
    montage_size: tuple[int, int] | None = None,
    quality_status: str = "not_evaluated",
    class_name_only_count: int = 0,
    class_conflict_count: int = 0,
    location_leak_count: int = 0,
    language_warning_count: int = 0,
    summary_contradiction: bool = False,
    semantic_warning_count: int = 0,
    class_name_only_detection_ids: tuple[int, ...] = (),
    class_conflict_detection_ids: tuple[int, ...] = (),
    location_leak_detection_ids: tuple[int, ...] = (),
    language_warning_detection_ids: tuple[int, ...] = (),
    exception_type: str = "",
    exception_message: str = "",
) -> dict[str, object]:
    """Convert one image run into a CSV row."""
    category = image_path.relative_to(input_dir).parts[0] if image_path.parent != input_dir else ""
    detections = yolo_result.detections if yolo_result is not None else []
    annotated_image_path = yolo_result.annotated_image_path if yolo_result is not None else None
    pipeline_status = pipeline_status or ("success" if status == "success" else "failed")
    montage_width = montage_size[0] if montage_size is not None else None
    montage_height = montage_size[1] if montage_size is not None else None
    full_image_width = full_image_size[0] if full_image_size is not None else None
    full_image_height = full_image_size[1] if full_image_size is not None else None
    return {
        "category": category,
        "image_name": image_path.name,
        "image_path": str(image_path),
        "ground_truth_class": ground_truth_for_category(category),
        "yolo_judgment": "NG" if detections else "OK",
        "yolo_detection_count": len(detections),
        "yolo_classes": "|".join(detection.class_name for detection in detections),
        "yolo_confidences": "|".join(f"{detection.confidence:.4f}" for detection in detections),
        "annotated_image_path": str(annotated_image_path) if annotated_image_path else "",
        "crop_montage_path": str(crop_montage_path) if crop_montage_path else "",
        "vlm_model": vlm_model,
        "vlm_response": vlm_response,
        "vlm_raw_response": vlm_raw_response,
        "vlm_parse_success": str(vlm_parse_success).lower(),
        "vlm_parse_error": vlm_parse_error,
        "vlm_fallback_used": str(vlm_fallback_used).lower(),
        "vlm_temperature": vlm_temperature,
        "vlm_top_p": vlm_top_p,
        "vlm_top_k": vlm_top_k,
        "vlm_repeat_penalty": vlm_repeat_penalty,
        "vlm_seed": vlm_seed,
        "vlm_image_mode": vlm_image_mode,
        "image_preparation_time_seconds": _format_seconds(image_preparation_seconds),
        "vlm_inference_time_seconds": _format_seconds(vlm_inference_seconds),
        "total_processing_time_seconds": _format_seconds(total_processing_seconds),
        "status": status,
        "image_status": image_status,
        "error_message": error_message,
        "pipeline_status": pipeline_status,
        "yolo_status": yolo_status,
        "vlm_status": vlm_status,
        "parse_status": parse_status,
        "fallback_used": _format_bool(vlm_fallback_used),
        "retry_count": retry_count,
        "failure_reason": failure_reason,
        "vlm_error_type": vlm_error_type,
        "vlm_error_message": vlm_error_message,
        "pipeline_success": _format_bool(status == "success"),
        "yolo_success": _format_bool(yolo_status == "success"),
        "vlm_attempted": _format_bool(vlm_status != "not_run"),
        "vlm_success": _format_bool(vlm_status in {"success", "retry_success"} and not vlm_fallback_used),
        "result_saved": _format_bool(result_saved),
        "http_status": _format_csv_value(ollama_metadata.http_status if ollama_metadata else None),
        "ollama_endpoint": ollama_metadata.endpoint if ollama_metadata else "",
        "ollama_stream": _format_bool(ollama_metadata.stream if ollama_metadata else None),
        "ollama_error": ollama_metadata.error if ollama_metadata and ollama_metadata.error else "",
        "ollama_done": _format_bool(ollama_metadata.done if ollama_metadata else None),
        "ollama_done_reason": ollama_metadata.done_reason if ollama_metadata and ollama_metadata.done_reason else "",
        "ollama_content_length": _format_csv_value(
            ollama_metadata.content_length if ollama_metadata else None
        ),
        "ollama_prompt_eval_count": _format_csv_value(
            ollama_metadata.prompt_eval_count if ollama_metadata else None
        ),
        "ollama_eval_count": _format_csv_value(ollama_metadata.eval_count if ollama_metadata else None),
        "ollama_total_duration": _format_csv_value(
            ollama_metadata.total_duration if ollama_metadata else None
        ),
        "ollama_load_duration": _format_csv_value(
            ollama_metadata.load_duration if ollama_metadata else None
        ),
        "ollama_prompt_eval_duration": _format_csv_value(
            ollama_metadata.prompt_eval_duration if ollama_metadata else None
        ),
        "ollama_eval_duration": _format_csv_value(
            ollama_metadata.eval_duration if ollama_metadata else None
        ),
        "vlm_image_count": _format_csv_value(vlm_image_count),
        "crop_count": _format_csv_value(crop_count),
        "vlm_full_image_size_limit": _format_csv_value(full_image_size_limit),
        "vlm_montage_size_limit": _format_csv_value(montage_size_limit),
        "vlm_full_image_width": _format_csv_value(full_image_width),
        "vlm_full_image_height": _format_csv_value(full_image_height),
        "montage_width": _format_csv_value(montage_width),
        "montage_height": _format_csv_value(montage_height),
        "quality_status": quality_status,
        "class_name_only_count": class_name_only_count,
        "class_conflict_count": class_conflict_count,
        "location_leak_count": location_leak_count,
        "language_warning_count": language_warning_count,
        "summary_contradiction": _format_bool(summary_contradiction),
        "semantic_warning_count": semantic_warning_count,
        "class_name_only_detection_ids": "|".join(
            str(detection_id) for detection_id in class_name_only_detection_ids
        ),
        "class_conflict_detection_ids": "|".join(
            str(detection_id) for detection_id in class_conflict_detection_ids
        ),
        "location_leak_detection_ids": "|".join(
            str(detection_id) for detection_id in location_leak_detection_ids
        ),
        "language_warning_detection_ids": "|".join(
            str(detection_id) for detection_id in language_warning_detection_ids
        ),
        "exception_type": exception_type,
        "exception_message": exception_message,
    }


def write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    """Write Excel-friendly UTF-8 CSV output."""
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    """Write UTF-8 JSON with stable non-destructive parent creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_image_result_json(path: Path, row: dict[str, object], yolo_result: YoloResult | None) -> None:
    """Persist one image result immediately as JSON."""
    detections = []
    if yolo_result is not None:
        detections = [
            {
                "detection_id": index,
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "bbox": [detection.x1, detection.y1, detection.x2, detection.y2],
                "location": detection.location,
            }
            for index, detection in enumerate(yolo_result.detections, start=1)
        ]
    write_json(path, {"result": row, "detections": detections})


def write_raw_response_if_needed(
    *,
    raw_responses_dir: Path,
    image_path: Path,
    raw_response: str,
    should_save: bool,
) -> Path | None:
    """Save raw VLM text only when debugging a failure/fallback path."""
    if not should_save or not raw_response:
        return None
    output_path = raw_responses_dir / f"{_safe_json_stem(image_path)}_failed.txt"
    output_path.write_text(raw_response, encoding="utf-8")
    return output_path


def _failure_diagnostic_text(vlm_service: object, raw_response: str) -> str:
    client = getattr(vlm_service, "client", None)
    metadata = getattr(vlm_service, "last_ollama_metadata", None)
    response_body = getattr(client, "last_response_data", None)
    if response_body is None and raw_response:
        response_body = raw_response
    diagnostic = {
        "http_status": metadata.http_status if metadata else None,
        "endpoint": metadata.endpoint if metadata else getattr(client, "endpoint", ""),
        "stream": metadata.stream if metadata else getattr(client, "stream", None),
        "done": metadata.done if metadata else None,
        "done_reason": metadata.done_reason if metadata else None,
        "error": metadata.error if metadata else None,
        "content_length": metadata.content_length if metadata else None,
        "prompt_eval_count": metadata.prompt_eval_count if metadata else None,
        "eval_count": metadata.eval_count if metadata else None,
        "total_duration": metadata.total_duration if metadata else None,
        "load_duration": metadata.load_duration if metadata else None,
        "prompt_eval_duration": metadata.prompt_eval_duration if metadata else None,
        "eval_duration": metadata.eval_duration if metadata else None,
        "response_body": response_body,
        "error_type": getattr(vlm_service, "last_error_type", "") or getattr(client, "last_error_type", ""),
        "error_message": getattr(vlm_service, "last_error_message", "") or getattr(client, "last_error_message", ""),
    }
    return json.dumps(diagnostic, ensure_ascii=False, indent=2)


def main() -> int:
    """Run the batch and keep processing after per-image failures."""
    configure_windows_console_encoding()
    args = parse_args()
    input_dir = _resolve_path(args.input_dir)
    output_dir = _resolve_path(args.output_dir)
    batch_paths = build_batch_paths(output_dir)
    args.save_crop_montage = True
    args.crop_montage_output_dir = str(batch_paths["montage"])
    csv_path = build_csv_path(output_dir)
    images = discover_images(input_dir)
    batch_started = perf_counter()

    print(f"[INFO] Found {len(images)} test images")
    if not images:
        write_csv(csv_path, [])
        write_json(output_dir / "batch_summary.json", _build_batch_summary([], [], perf_counter() - batch_started))
        write_json(output_dir / "failed_images.json", [])
        print("[INFO] Batch completed")
        print("[INFO] Total images: 0")
        print("[INFO] Success: 0")
        print("[INFO] Failed: 0")
        print(f"[INFO] Result CSV: {csv_path.resolve()}")
        return 0

    yolo_service = build_yolo_service(args)
    vlm_service = build_vlm_service(args)
    rows: list[dict[str, object]] = []
    image_summaries: list[dict[str, object]] = []
    failed_images: list[dict[str, object]] = []
    consecutive_zero_value_fallbacks = 0
    vlm_circuit_breaker_open = False
    category_counts = Counter(path.relative_to(input_dir).parts[0] for path in images)
    print(f"[INFO] VLM full image size limit: {args.vlm_full_image_size or args.vlm_image_size}")
    print(f"[INFO] VLM montage size limit: {args.vlm_montage_size or args.vlm_crop_montage_size}")
    print(f"[INFO] VLM image mode: {args.vlm_image_mode}")

    for index, image_path in enumerate(images, start=1):
        relative_path = image_path.relative_to(input_dir)
        prefix = f"[INFO] [{index}/{len(images)}] {relative_path}"
        print(f"{prefix} processing started")
        started = perf_counter()
        yolo_result: YoloResult | None = None
        result_json_path = batch_paths["results"] / f"{_safe_json_stem(image_path)}.json"
        try:
            print(f"{prefix} YOLO started")
            yolo_output_path = batch_paths["result_images"] / f"{_safe_json_stem(image_path)}_result{image_path.suffix}"
            yolo_result = yolo_service.detect(image_path, output_path=yolo_output_path)
            print(f"{prefix} YOLO completed")
            if yolo_result.defect_count == 0:
                print(f"{prefix} OK")
                print(f"{prefix} VLM skipped")
                vlm_response = SKIPPED_VLM_MESSAGE
                vlm_raw_response = ""
                vlm_parse_success = False
                vlm_parse_error = ""
                vlm_fallback_used = False
                crop_montage_path = None
                image_preparation_seconds = None
                vlm_inference_seconds = None
                vlm_status = "not_run"
                parse_status = "not_attempted"
                ollama_metadata = None
                vlm_image_count = None
                crop_count = None
                montage_size = None
                full_image_size = None
                full_image_size_limit = None
                montage_size_limit = None
                vlm_image_mode = ""
                quality_status = "not_evaluated"
                class_name_only_count = 0
                class_conflict_count = 0
                location_leak_count = 0
                language_warning_count = 0
                summary_contradiction = False
                semantic_warning_count = 0
                class_name_only_detection_ids = ()
                class_conflict_detection_ids = ()
                location_leak_detection_ids = ()
                language_warning_detection_ids = ()
                image_status = "completed"
                retry_count = 0
                failure_reason = ""
                vlm_error_type = ""
                vlm_error_message = ""
            else:
                print(f"{prefix} NG, detection {yolo_result.defect_count} count")
                if vlm_circuit_breaker_open:
                    print(f"{prefix} VLM skipped by circuit breaker")
                    vlm_response = format_yolo_fallback_response(yolo_result)
                    retry_count = 0
                    failure_reason = "circuit_breaker_open"
                    vlm_error_type = "circuit_breaker_open"
                    vlm_error_message = "VLM skipped after repeated zero-value Ollama responses."
                    crop_montage_path = None
                    image_preparation_seconds = None
                    vlm_inference_seconds = None
                    vlm_raw_response = ""
                    vlm_parse_success = False
                    vlm_parse_error = vlm_error_message
                    vlm_fallback_used = True
                    vlm_status = "circuit_breaker_open"
                    parse_status = "not_attempted"
                    ollama_metadata = None
                    vlm_image_count = None
                    vlm_image_mode = args.vlm_image_mode
                    crop_count = None
                    full_image_size_limit = None
                    montage_size_limit = None
                    full_image_size = None
                    montage_size = None
                    quality_status = "not_evaluated"
                    class_name_only_count = 0
                    class_conflict_count = 0
                    location_leak_count = 0
                    language_warning_count = 0
                    summary_contradiction = False
                    semantic_warning_count = 0
                    class_name_only_detection_ids = ()
                    class_conflict_detection_ids = ()
                    location_leak_detection_ids = ()
                    language_warning_detection_ids = ()
                    image_status = "completed_with_fallback"
                else:
                    print(f"{prefix} VLM started")
                    vlm_response = vlm_service.describe_defects(
                        yolo_result.annotated_image_path or image_path,
                        yolo_result,
                    ) or ""
                    retry_count = vlm_service.last_retry_count
                    failure_reason = vlm_service.last_failure_reason
                    vlm_error_type = getattr(vlm_service, "last_error_type", "")
                    vlm_error_message = getattr(vlm_service, "last_error_message", "")
                    if retry_count:
                        print(f"{prefix} VLM retry count: {retry_count}/{args.vlm_max_retries}")
                    info = vlm_service.last_preparation_info
                    crop_montage_path = info.crop_montage_path if info else None
                    image_preparation_seconds = info.image_preparation_seconds if info else None
                    vlm_inference_seconds = info.inference_seconds if info else None
                    vlm_raw_response = vlm_service.last_raw_response or ""
                    vlm_parse_success = vlm_service.last_parse_success
                    vlm_parse_error = vlm_service.last_parse_error
                    vlm_fallback_used = vlm_service.last_fallback_used
                    vlm_status = vlm_service.last_vlm_status
                    parse_status = vlm_service.last_parse_status
                    ollama_metadata = vlm_service.last_ollama_metadata
                    vlm_image_count = info.image_count if info else None
                    vlm_image_mode = info.image_mode if info else ""
                    crop_count = info.detection_crop_count if info else None
                    full_image_size_limit = info.full_image_size_limit if info else None
                    montage_size_limit = info.crop_montage_size_limit if info else None
                    full_image_size = info.full_image_size if info else None
                    montage_size = info.crop_montage_size if info else None
                    quality = vlm_service.last_quality_info
                    quality_status = quality.quality_status
                    class_name_only_count = quality.class_name_only_count
                    class_conflict_count = quality.class_conflict_count
                    location_leak_count = quality.location_leak_count
                    language_warning_count = quality.language_warning_count
                    summary_contradiction = quality.summary_contradiction
                    semantic_warning_count = quality.semantic_warning_count
                    class_name_only_detection_ids = quality.class_name_only_detection_ids
                    class_conflict_detection_ids = quality.class_conflict_detection_ids
                    location_leak_detection_ids = quality.location_leak_detection_ids
                    language_warning_detection_ids = quality.language_warning_detection_ids
                    image_status = "completed_with_fallback" if vlm_fallback_used else "completed"
                    if full_image_size is not None:
                        print(f"[INFO] VLM full image size: {full_image_size[0]}x{full_image_size[1]}")
                    if montage_size is not None:
                        print(f"[INFO] VLM crop montage size: {montage_size[0]}x{montage_size[1]}")
                    if info and info.request_json_size is not None:
                        print(f"[INFO] VLM request JSON size: {info.request_json_size}")
                    if info and info.zero_value_recovery_used:
                        print(f"[INFO] VLM zero-value recovery used: true")
                        print(f"[INFO] VLM zero-value retry image size: {info.zero_value_recovery_image_size}")
                        print(f"[INFO] VLM zero-value unload succeeded: {str(info.zero_value_unload_succeeded).lower()}")
                    print(f"[INFO] VLM image mode: {vlm_image_mode}")
                    print(f"[INFO] VLM image count: {vlm_image_count}")
                    if vlm_inference_seconds is not None:
                        print(f"[INFO] VLM inference time: {vlm_inference_seconds:.3f}s")
                    print(f"[INFO] VLM parse success: {str(vlm_parse_success).lower()}")
                    print(f"[INFO] VLM fallback used: {str(vlm_fallback_used).lower()}")
                    print(f"[INFO] VLM status: {vlm_status}")
                    print(f"[INFO] VLM parse status: {parse_status}")
                    print(f"[INFO] VLM quality status: {quality_status}")
                    print(f"[INFO] Class-name-only visual features: {class_name_only_count}")
                    print(f"[INFO] Class-conflict visual features: {class_conflict_count}")
                    print(f"[INFO] Location-leak visual features: {location_leak_count}")
                    print(f"[INFO] Language warning visual features: {language_warning_count}")
                    print(f"[INFO] Summary contradiction: {str(summary_contradiction).lower()}")
                    print(f"[INFO] Semantic warning count: {semantic_warning_count}")
                    if vlm_fallback_used:
                        print(f"{prefix} VLM final failure, fallback used")
                    else:
                        print(f"{prefix} VLM response validation completed")

                    if vlm_fallback_used and _row_has_failure(
                        {"vlm_status": vlm_status, "parse_status": parse_status, "failure_reason": failure_reason},
                        "done_false",
                    ):
                        consecutive_zero_value_fallbacks += 1
                    else:
                        consecutive_zero_value_fallbacks = 0
                    if (
                        args.vlm_circuit_breaker_threshold > 0
                        and consecutive_zero_value_fallbacks >= args.vlm_circuit_breaker_threshold
                    ):
                        vlm_circuit_breaker_open = True
                        print(
                            "[WARN] VLM circuit breaker opened after "
                            f"{consecutive_zero_value_fallbacks} consecutive zero-value fallbacks"
                        )

            row = result_to_row(
                image_path=image_path,
                input_dir=input_dir,
                yolo_result=yolo_result,
                vlm_model=args.vlm_model,
                vlm_response=vlm_response,
                vlm_raw_response=vlm_raw_response,
                vlm_parse_success=vlm_parse_success,
                vlm_parse_error=vlm_parse_error,
                vlm_fallback_used=vlm_fallback_used,
                vlm_temperature=args.vlm_temperature,
                vlm_top_p=args.vlm_top_p,
                vlm_top_k=args.vlm_top_k,
                vlm_repeat_penalty=args.vlm_repeat_penalty,
                vlm_seed=args.vlm_seed,
                vlm_image_mode=vlm_image_mode,
                crop_montage_path=crop_montage_path,
                image_preparation_seconds=image_preparation_seconds,
                vlm_inference_seconds=vlm_inference_seconds,
                total_processing_seconds=perf_counter() - started,
                status="success",
                image_status=image_status,
                pipeline_status="success",
                yolo_status="success",
                vlm_status=vlm_status,
                parse_status=parse_status,
                retry_count=retry_count,
                failure_reason=failure_reason,
                vlm_error_type=vlm_error_type,
                vlm_error_message=vlm_error_message,
                ollama_metadata=ollama_metadata,
                vlm_image_count=vlm_image_count,
                crop_count=crop_count,
                full_image_size_limit=full_image_size_limit,
                montage_size_limit=montage_size_limit,
                full_image_size=full_image_size,
                montage_size=montage_size,
                quality_status=quality_status,
                class_name_only_count=class_name_only_count,
                class_conflict_count=class_conflict_count,
                location_leak_count=location_leak_count,
                language_warning_count=language_warning_count,
                summary_contradiction=summary_contradiction,
                semantic_warning_count=semantic_warning_count,
                class_name_only_detection_ids=class_name_only_detection_ids,
                class_conflict_detection_ids=class_conflict_detection_ids,
                location_leak_detection_ids=location_leak_detection_ids,
                language_warning_detection_ids=language_warning_detection_ids,
            )
            rows.append(row)
            write_image_result_json(result_json_path, row, yolo_result)
            write_raw_response_if_needed(
                raw_responses_dir=batch_paths["raw_responses"],
                image_path=image_path,
                raw_response=_failure_diagnostic_text(vlm_service, vlm_raw_response),
                should_save=args.save_raw_response_on_failure and bool(vlm_fallback_used),
            )
            write_csv(csv_path, rows)
            image_summaries.append(_summary_for_row(row))
            print(f"{prefix} result saved")
            print(f"{prefix} processing completed")
        except Exception as exc:
            row = result_to_row(
                image_path=image_path,
                input_dir=input_dir,
                yolo_result=yolo_result,
                vlm_model=args.vlm_model,
                vlm_response="",
                vlm_raw_response=vlm_service.last_raw_response or "",
                vlm_parse_success=vlm_service.last_parse_success,
                vlm_parse_error=vlm_service.last_parse_error,
                vlm_fallback_used=vlm_service.last_fallback_used,
                vlm_temperature=args.vlm_temperature,
                vlm_top_p=args.vlm_top_p,
                vlm_top_k=args.vlm_top_k,
                vlm_repeat_penalty=args.vlm_repeat_penalty,
                vlm_seed=args.vlm_seed,
                vlm_image_mode=(
                    vlm_service.last_preparation_info.image_mode
                    if vlm_service.last_preparation_info
                    else ""
                ),
                crop_montage_path=None,
                image_preparation_seconds=None,
                vlm_inference_seconds=None,
                total_processing_seconds=perf_counter() - started,
                status="error",
                image_status="failed",
                error_message=str(exc),
                pipeline_status="failed",
                yolo_status="failed" if yolo_result is None else "success",
                vlm_status=vlm_service.last_vlm_status if yolo_result is not None else "not_run",
                parse_status=vlm_service.last_parse_status if yolo_result is not None else "not_attempted",
                retry_count=vlm_service.last_retry_count if yolo_result is not None else 0,
                failure_reason=vlm_service.last_failure_reason if yolo_result is not None else type(exc).__name__,
                vlm_error_type=(
                    getattr(vlm_service, "last_error_type", "") if yolo_result is not None else type(exc).__name__
                ),
                vlm_error_message=(
                    getattr(vlm_service, "last_error_message", "") if yolo_result is not None else str(exc)
                ),
                ollama_metadata=vlm_service.last_ollama_metadata,
                quality_status=vlm_service.last_quality_info.quality_status,
                class_name_only_count=vlm_service.last_quality_info.class_name_only_count,
                class_conflict_count=vlm_service.last_quality_info.class_conflict_count,
                location_leak_count=vlm_service.last_quality_info.location_leak_count,
                language_warning_count=vlm_service.last_quality_info.language_warning_count,
                summary_contradiction=vlm_service.last_quality_info.summary_contradiction,
                semantic_warning_count=vlm_service.last_quality_info.semantic_warning_count,
                class_name_only_detection_ids=(
                    vlm_service.last_quality_info.class_name_only_detection_ids
                ),
                class_conflict_detection_ids=(
                    vlm_service.last_quality_info.class_conflict_detection_ids
                ),
                location_leak_detection_ids=(
                    vlm_service.last_quality_info.location_leak_detection_ids
                ),
                language_warning_detection_ids=(
                    vlm_service.last_quality_info.language_warning_detection_ids
                ),
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            rows.append(row)
            failed_images.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            write_image_result_json(result_json_path, row, yolo_result)
            write_raw_response_if_needed(
                raw_responses_dir=batch_paths["raw_responses"],
                image_path=image_path,
                raw_response=_failure_diagnostic_text(vlm_service, vlm_service.last_raw_response or ""),
                should_save=args.save_raw_response_on_failure,
            )
            write_csv(csv_path, rows)
            image_summaries.append(_summary_for_row(row))
            print(f"[ERROR] [{index}/{len(images)}] {relative_path} completed with error: {exc}")
            if not args.continue_on_error:
                break

    write_csv(csv_path, rows)
    batch_summary = _build_batch_summary(rows, image_summaries, perf_counter() - batch_started)
    write_json(output_dir / "batch_summary.json", batch_summary)
    write_json(output_dir / "failed_images.json", failed_images)
    success_count = sum(1 for row in rows if row["status"] == "success")
    failed_count = len(rows) - success_count
    print("[INFO] Batch completed")
    print(f"[INFO] Total images: {len(rows)}")
    print(f"[INFO] Pipeline completed: {success_count}")
    print(f"[INFO] Pipeline failed: {failed_count}")
    print(f"[INFO] YOLO success: {batch_summary['yolo_success_count']}")
    print(f"[INFO] YOLO failed: {batch_summary['yolo_failed_count']}")
    print(f"[INFO] VLM success: {batch_summary['vlm_success_count']}")
    print(f"[INFO] VLM fallback: {batch_summary['fallback_used_count']}")
    print(f"[INFO] VLM skipped for OK: {batch_summary['vlm_skipped_count']}")
    print(f"[INFO] Result save success: {batch_summary['result_save_success_count']}")
    for category, count in sorted(category_counts.items()):
        print(f"[INFO] Category {category}: {count}")
    print(f"[INFO] Result CSV: {csv_path.resolve()}")
    return 0 if failed_count == 0 else 1


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _format_seconds(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _format_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return str(value).lower()


def _format_csv_value(value: object | None) -> object:
    return "" if value is None else value


def _safe_json_stem(image_path: Path) -> str:
    relative_parts = image_path.with_suffix("").parts[-3:]
    safe = "_".join(relative_parts)
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in safe)
    return safe or "image"


def _summary_for_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "image_name": row["image_name"],
        "yolo_judgment": row["yolo_judgment"],
        "detection_count": row["yolo_detection_count"],
        "vlm_executed": row["vlm_status"] != "not_run",
        "vlm_status": row["vlm_status"],
        "vlm_success": row.get("vlm_success") == "true",
        "retry_count": row["retry_count"],
        "fallback_used": row["fallback_used"] == "true",
        "vlm_error_type": row.get("vlm_error_type", ""),
        "vlm_error_message": row.get("vlm_error_message", ""),
        "processing_time_ms": _seconds_string_to_ms(row["total_processing_time_seconds"]),
        "status": row["image_status"],
    }


def _build_batch_summary(
    rows: list[dict[str, object]],
    image_summaries: list[dict[str, object]],
    total_processing_seconds: float,
) -> dict[str, object]:
    total_images = len(rows)
    vlm_rows = [row for row in rows if row["vlm_status"] != "not_run"]
    processing_seconds = [
        float(row["total_processing_time_seconds"])
        for row in rows
        if str(row["total_processing_time_seconds"]).strip()
    ]
    vlm_seconds = [
        float(row["vlm_inference_time_seconds"])
        for row in rows
        if str(row["vlm_inference_time_seconds"]).strip()
    ]
    return {
        "total_images": total_images,
        "completed_count": sum(1 for row in rows if row["status"] == "success"),
        "pipeline_completed_count": sum(1 for row in rows if _row_bool(row, "pipeline_success", row["status"] == "success")),
        "pipeline_failed_count": sum(1 for row in rows if not _row_bool(row, "pipeline_success", row["status"] == "success")),
        "yolo_success_count": sum(1 for row in rows if _row_bool(row, "yolo_success", row.get("yolo_status") == "success")),
        "yolo_failed_count": sum(1 for row in rows if not _row_bool(row, "yolo_success", row.get("yolo_status") == "success")),
        "ok_image_count": sum(1 for row in rows if row["yolo_judgment"] == "OK"),
        "ng_image_count": sum(1 for row in rows if row["yolo_judgment"] == "NG"),
        "vlm_executed_count": len(vlm_rows),
        "vlm_attempted_count": sum(1 for row in rows if _row_bool(row, "vlm_attempted", row["vlm_status"] != "not_run")),
        "vlm_success_count": sum(
            1
            for row in rows
            if _row_bool(
                row,
                "vlm_success",
                row["vlm_status"] in {"success", "retry_success"} and row["fallback_used"] != "true",
            )
        ),
        "vlm_skipped_count": sum(1 for row in rows if row["vlm_status"] == "not_run"),
        "vlm_first_success_count": sum(1 for row in rows if row["vlm_status"] == "success"),
        "vlm_retry_success_count": sum(1 for row in rows if row["vlm_status"] == "retry_success"),
        "fallback_used_count": sum(1 for row in rows if row["fallback_used"] == "true"),
        "result_save_success_count": sum(1 for row in rows if _row_bool(row, "result_saved", True)),
        "final_failed_count": sum(1 for row in rows if row["status"] != "success"),
        "done_false_count": sum(1 for row in rows if _row_has_failure(row, "done_false")),
        "empty_content_count": sum(1 for row in rows if _row_has_failure(row, "empty_content")),
        "invalid_json_count": sum(1 for row in rows if _row_has_failure(row, "json_parse_failed")),
        "schema_error_count": sum(1 for row in rows if _row_has_failure(row, "validation_failed")),
        "timeout_count": sum(1 for row in rows if "timeout" in str(row["failure_reason"]).lower()),
        "total_processing_time_seconds": round(total_processing_seconds, 3),
        "average_image_processing_time_seconds": _average(processing_seconds),
        "average_vlm_processing_time_seconds": _average(vlm_seconds),
        "images": image_summaries,
    }


def _seconds_string_to_ms(value: object) -> int:
    try:
        return int(round(float(value) * 1000))
    except (TypeError, ValueError):
        return 0


def _row_has_failure(row: dict[str, object], failure_name: str) -> bool:
    return row["vlm_status"] == failure_name or row["parse_status"] == failure_name or failure_name in str(
        row["failure_reason"]
    ).split("|")


def _row_bool(row: dict[str, object], key: str, default: bool) -> bool:
    value = row.get(key)
    if value in {"true", "false"}:
        return value == "true"
    if isinstance(value, bool):
        return value
    return default


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


if __name__ == "__main__":
    raise SystemExit(main())
