from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.console_encoding import configure_windows_console_encoding
from service.inspection_service import InspectionService
from service.vlm_service import VlmService
from service.yolo_service import YoloService
from vlm.prompt_builder import PromptBuilder
from vlm.response_parser import VlmResponseParser
from vlm.vlm_client import VlmClient
from yolo.detector import YoloDetector
from yolo.model_loader import YoloModelLoader
from yolo.yolo_config import YoloConfig


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be at least 0")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse terminal inspection options."""
    parser = argparse.ArgumentParser(
        description="Run YOLO PCB inspection and optional Ollama VLM explanation."
    )
    parser.add_argument("--image", required=True, help="Input PCB image path.")
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
        help="Save the generated detection crop montage image for debugging.",
    )
    parser.add_argument(
        "--crop-montage-output-dir",
        default="data/result_images/montage",
        help="Directory used when --save-crop-montage is enabled.",
    )
    parser.add_argument("--vlm-max-retries", type=non_negative_int, default=2, help="Maximum VLM retry count.")
    parser.add_argument("--vlm-retry-delay", type=float, default=0.5, help="Delay between VLM retries in seconds.")
    parser.add_argument("--vlm-timeout", type=float, default=120.0, help="Ollama HTTP timeout in seconds.")
    parser.add_argument("--skip-vlm", action="store_true", help="Run YOLO only and skip VLM explanation.")
    parser.add_argument("--debug-vlm", action="store_true", help="Print the raw VLM response after the sanitized explanation.")
    return parser.parse_args()


def build_yolo_service(args: argparse.Namespace) -> YoloService:
    """Create a configured YOLO service for one CLI run."""
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    config = YoloConfig(
        model_path=model_path,
        confidence_threshold=args.conf,
        image_size=args.imgsz,
        iou_threshold=args.iou,
        device=str(args.device),
    )
    loader = YoloModelLoader(config)
    detector = YoloDetector(model_loader=loader, config=config)
    return YoloService(detector)


def build_vlm_service(args: argparse.Namespace) -> VlmService:
    """Create an Ollama-backed VLM service."""
    full_image_size = args.vlm_full_image_size or args.vlm_image_size
    montage_size = args.vlm_montage_size or args.vlm_crop_montage_size
    client = VlmClient(
        model_name=args.vlm_model,
        host=args.ollama_host,
        temperature=args.vlm_temperature,
        top_p=args.vlm_top_p,
        top_k=args.vlm_top_k,
        repeat_penalty=args.vlm_repeat_penalty,
        seed=args.vlm_seed,
        num_ctx=args.vlm_num_ctx,
        num_predict=args.vlm_num_predict,
        debug_response=args.vlm_debug_response,
        timeout_seconds=getattr(args, "vlm_timeout", 120.0),
    )
    return VlmService(
        client=client,
        prompt_builder=PromptBuilder(),
        response_parser=VlmResponseParser(),
        image_size=full_image_size,
        image_quality=args.vlm_image_quality,
        crop_montage_size=montage_size,
        crop_padding=args.vlm_crop_padding,
        crop_min_size=args.vlm_crop_min_size,
        crop_max_size=args.vlm_crop_max_size,
        save_crop_montage=args.save_crop_montage,
        crop_montage_output_dir=PROJECT_ROOT / args.crop_montage_output_dir,
        image_mode=args.vlm_image_mode,
        max_retries=getattr(args, "vlm_max_retries", 0),
        retry_delay_seconds=getattr(args, "vlm_retry_delay", 0.0),
    )


def print_detection_rows(detections: object) -> None:
    """Print detection rows in a compact terminal-friendly format."""
    print()
    print("Detection results")
    if not detections:
        print("(none)")
        return

    for index, detection in enumerate(detections, start=1):
        print(
            f"{index}. {detection.class_name} | confidence={detection.confidence:.4f} | "
            f"location={detection.location or '위치 미계산'} | "
            f"box=({detection.x1}, {detection.y1}, {detection.x2}, {detection.y2})"
        )


def main() -> int:
    """Run YOLO and optionally VLM from the terminal."""
    configure_windows_console_encoding()
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    try:
        yolo_service = build_yolo_service(args)
        print(f"[INFO] Input image: {image_path}")
        print(f"[INFO] YOLO model: {Path(args.model)}")

        if args.skip_vlm:
            yolo_result = yolo_service.detect(image_path)
            status = "NG" if yolo_result.is_ng else "OK"
            print(f"[INFO] Judgment: {status}")
            print(f"[INFO] Detection count: {yolo_result.defect_count}")
            print(f"[INFO] Result image: {yolo_result.annotated_image_path}")
            if yolo_result.defect_count == 0:
                print("[INFO] VLM analysis skipped")
            else:
                print("[INFO] VLM analysis skipped by --skip-vlm")
            print_detection_rows(yolo_result.detections)
            return 0

        print(f"[INFO] Ollama host: {args.ollama_host}")
        print(f"[INFO] VLM model: {args.vlm_model}")
        print(f"[INFO] VLM num_ctx: {args.vlm_num_ctx}")
        print(f"[INFO] VLM num_predict: {args.vlm_num_predict}")
        print(f"[INFO] VLM temperature: {args.vlm_temperature}")
        print(f"[INFO] VLM top_p: {args.vlm_top_p}")
        print(f"[INFO] VLM top_k: {args.vlm_top_k}")
        print(f"[INFO] VLM repeat_penalty: {args.vlm_repeat_penalty}")
        print(f"[INFO] VLM seed: {args.vlm_seed}")
        print(f"[INFO] VLM max retries: {args.vlm_max_retries}")
        print(f"[INFO] VLM retry delay: {args.vlm_retry_delay}")
        print(f"[INFO] VLM timeout: {args.vlm_timeout}")
        print(f"[INFO] VLM image mode: {args.vlm_image_mode}")
        print(f"[INFO] VLM full image size limit: {args.vlm_full_image_size or args.vlm_image_size}")
        print(f"[INFO] VLM montage size limit: {args.vlm_montage_size or args.vlm_crop_montage_size}")
        print(f"[INFO] VLM crop padding: {args.vlm_crop_padding}")
        if args.save_crop_montage:
            print(f"[INFO] Crop montage output dir: {PROJECT_ROOT / args.crop_montage_output_dir}")

        vlm_service = build_vlm_service(args)
        inspection_service = InspectionService(
            yolo_service=yolo_service,
            vlm_service=vlm_service,
        )
        result = inspection_service.inspect(image_path)
        print(f"[INFO] Judgment: {result.status}")
        print(f"[INFO] Detection count: {result.defect_count}")
        print(f"[INFO] Result image: {result.result_image_path}")
        if result.defect_count == 0:
            print("[INFO] VLM analysis skipped")
        elif vlm_service.last_preparation_info is not None:
            info = vlm_service.last_preparation_info
            print(f"[INFO] VLM image mode: {info.image_mode}")
            print(f"[INFO] VLM image count: {info.image_count}")
            print(f"[INFO] VLM full image prepared: {str(info.full_image_prepared).lower()}")
            print(f"[INFO] VLM crop montage prepared: {str(info.crop_montage_prepared).lower()}")
            print(f"[INFO] VLM detection crop count: {info.detection_crop_count}")
            if info.full_image_size is not None:
                width, height = info.full_image_size
                print(f"[INFO] VLM full image size: {width}x{height}")
            if info.crop_montage_size is not None:
                width, height = info.crop_montage_size
                print(f"[INFO] VLM crop montage size: {width}x{height}")
            if info.crop_montage_path is not None:
                print(f"[INFO] Crop montage path: {info.crop_montage_path.resolve()}")
            if info.image_preparation_seconds is not None:
                print(f"[INFO] VLM image preparation time: {info.image_preparation_seconds:.3f}s")
            if info.inference_seconds is not None:
                print(f"[INFO] VLM inference time: {info.inference_seconds:.3f}s")
            print(f"[INFO] VLM parse success: {str(vlm_service.last_parse_success).lower()}")
            print(f"[INFO] VLM fallback used: {str(vlm_service.last_fallback_used).lower()}")
            print(f"[INFO] VLM status: {vlm_service.last_vlm_status}")
            print(f"[INFO] VLM parse status: {vlm_service.last_parse_status}")
            quality = vlm_service.last_quality_info
            print(f"[INFO] VLM quality status: {quality.quality_status}")
            print(f"[INFO] Class-name-only visual features: {quality.class_name_only_count}")
            print(f"[INFO] Summary contradiction: {str(quality.summary_contradiction).lower()}")
            print(f"[INFO] Semantic warning count: {quality.semantic_warning_count}")
            metadata = vlm_service.last_ollama_metadata
            if metadata is not None:
                print(f"[INFO] Ollama done: {metadata.done}")
                print(f"[INFO] Ollama done reason: {metadata.done_reason}")
                print(f"[INFO] Ollama content length: {metadata.content_length}")
                print(f"[INFO] Ollama prompt eval count: {metadata.prompt_eval_count}")
                print(f"[INFO] Ollama eval count: {metadata.eval_count}")
                print(f"[INFO] Ollama total duration: {metadata.total_duration}")
            if vlm_service.last_parse_error:
                print(f"[INFO] VLM parse error: {vlm_service.last_parse_error}")
        print_detection_rows(result.detections)
        if result.vlm_explanation:
            print()
            print("[VLM 설명]")
            print(result.vlm_explanation)
        if args.debug_vlm and vlm_service.last_raw_response:
            print()
            print("[VLM raw response]")
            print(vlm_service.last_raw_response)

        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
