from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.yolo_result import YoloResult
from scripts.test_yolo_vlm import build_vlm_service, build_yolo_service, positive_int
from vlm.ollama_response import OllamaResponseMetadata

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
    "error_message",
    "pipeline_status",
    "yolo_status",
    "vlm_status",
    "parse_status",
    "fallback_used",
    "http_status",
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
    "summary_contradiction",
    "semantic_warning_count",
    "class_name_only_detection_ids",
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
    ollama_metadata: OllamaResponseMetadata | None = None,
    vlm_image_count: int | None = None,
    crop_count: int | None = None,
    full_image_size_limit: int | None = None,
    montage_size_limit: int | None = None,
    full_image_size: tuple[int, int] | None = None,
    montage_size: tuple[int, int] | None = None,
    quality_status: str = "not_evaluated",
    class_name_only_count: int = 0,
    summary_contradiction: bool = False,
    semantic_warning_count: int = 0,
    class_name_only_detection_ids: tuple[int, ...] = (),
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
        "error_message": error_message,
        "pipeline_status": pipeline_status,
        "yolo_status": yolo_status,
        "vlm_status": vlm_status,
        "parse_status": parse_status,
        "fallback_used": _format_bool(vlm_fallback_used),
        "http_status": _format_csv_value(ollama_metadata.http_status if ollama_metadata else None),
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
        "summary_contradiction": _format_bool(summary_contradiction),
        "semantic_warning_count": semantic_warning_count,
        "class_name_only_detection_ids": "|".join(
            str(detection_id) for detection_id in class_name_only_detection_ids
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


def main() -> int:
    """Run the batch and keep processing after per-image failures."""
    args = parse_args()
    input_dir = _resolve_path(args.input_dir)
    output_dir = _resolve_path(args.output_dir)
    csv_path = build_csv_path(output_dir)
    images = discover_images(input_dir)

    print(f"[INFO] Found {len(images)} test images")
    if not images:
        write_csv(csv_path, [])
        print("[INFO] Batch completed")
        print("[INFO] Total images: 0")
        print("[INFO] Success: 0")
        print("[INFO] Failed: 0")
        print(f"[INFO] Result CSV: {csv_path.resolve()}")
        return 0

    yolo_service = build_yolo_service(args)
    vlm_service = build_vlm_service(args)
    rows: list[dict[str, object]] = []
    category_counts = Counter(path.relative_to(input_dir).parts[0] for path in images)
    print(f"[INFO] VLM full image size limit: {args.vlm_full_image_size or args.vlm_image_size}")
    print(f"[INFO] VLM montage size limit: {args.vlm_montage_size or args.vlm_crop_montage_size}")
    print(f"[INFO] VLM image mode: {args.vlm_image_mode}")

    for index, image_path in enumerate(images, start=1):
        relative_path = image_path.relative_to(input_dir)
        print(f"[INFO] [{index}/{len(images)}] Processing {relative_path}")
        started = perf_counter()
        yolo_result: YoloResult | None = None
        try:
            yolo_result = yolo_service.detect(image_path)
            print(f"[INFO] YOLO detections: {yolo_result.defect_count}")
            if yolo_result.defect_count == 0:
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
                summary_contradiction = False
                semantic_warning_count = 0
                class_name_only_detection_ids = ()
            else:
                vlm_response = vlm_service.describe_defects(
                    yolo_result.annotated_image_path or image_path,
                    yolo_result,
                ) or ""
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
                summary_contradiction = quality.summary_contradiction
                semantic_warning_count = quality.semantic_warning_count
                class_name_only_detection_ids = quality.class_name_only_detection_ids
                if full_image_size is not None:
                    print(f"[INFO] VLM full image size: {full_image_size[0]}x{full_image_size[1]}")
                if montage_size is not None:
                    print(f"[INFO] VLM crop montage size: {montage_size[0]}x{montage_size[1]}")
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
                print(f"[INFO] Summary contradiction: {str(summary_contradiction).lower()}")
                print(f"[INFO] Semantic warning count: {semantic_warning_count}")

            rows.append(
                result_to_row(
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
                    pipeline_status="success",
                    yolo_status="success",
                    vlm_status=vlm_status,
                    parse_status=parse_status,
                    ollama_metadata=ollama_metadata,
                    vlm_image_count=vlm_image_count,
                    crop_count=crop_count,
                    full_image_size_limit=full_image_size_limit,
                    montage_size_limit=montage_size_limit,
                    full_image_size=full_image_size,
                    montage_size=montage_size,
                    quality_status=quality_status,
                    class_name_only_count=class_name_only_count,
                    summary_contradiction=summary_contradiction,
                    semantic_warning_count=semantic_warning_count,
                    class_name_only_detection_ids=class_name_only_detection_ids,
                )
            )
            print("[INFO] Completed: success")
        except Exception as exc:
            rows.append(
                result_to_row(
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
                    error_message=str(exc),
                    pipeline_status="failed",
                    yolo_status="failed" if yolo_result is None else "success",
                    vlm_status=vlm_service.last_vlm_status if yolo_result is not None else "not_run",
                    parse_status=vlm_service.last_parse_status if yolo_result is not None else "not_attempted",
                    ollama_metadata=vlm_service.last_ollama_metadata,
                    quality_status=vlm_service.last_quality_info.quality_status,
                    class_name_only_count=vlm_service.last_quality_info.class_name_only_count,
                    summary_contradiction=vlm_service.last_quality_info.summary_contradiction,
                    semantic_warning_count=vlm_service.last_quality_info.semantic_warning_count,
                    class_name_only_detection_ids=(
                        vlm_service.last_quality_info.class_name_only_detection_ids
                    ),
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
            )
            print(f"[ERROR] Completed: error: {exc}")

    write_csv(csv_path, rows)
    success_count = sum(1 for row in rows if row["status"] == "success")
    failed_count = len(rows) - success_count
    print("[INFO] Batch completed")
    print(f"[INFO] Total images: {len(rows)}")
    print(f"[INFO] Success: {success_count}")
    print(f"[INFO] Failed: {failed_count}")
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


if __name__ == "__main__":
    raise SystemExit(main())
