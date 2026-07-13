from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.inspection_service import InspectionService
from service.vlm_service import VlmService
from service.yolo_service import YoloService
from vlm.prompt_builder import PromptBuilder
from vlm.response_parser import VlmResponseParser
from vlm.vlm_client import VlmClient
from yolo.detector import YoloDetector
from yolo.model_loader import YoloModelLoader
from yolo.yolo_config import YoloConfig


def parse_args() -> argparse.Namespace:
    """Parse terminal inspection options."""
    parser = argparse.ArgumentParser(description="Run YOLO PCB inspection and optional Ollama VLM explanation.")
    parser.add_argument("--image", required=True, help="Input PCB image path.")
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--device", default="0", help="YOLO device, for example 0 or cpu.")
    parser.add_argument("--vlm-model", default="qwen2.5vl:3b", help="Ollama VLM model name.")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434", help="Ollama host URL.")
    parser.add_argument("--vlm-num-ctx", type=int, default=8192, help="Ollama VLM context window.")
    parser.add_argument("--vlm-num-predict", type=int, default=512, help="Ollama VLM max generated tokens.")
    parser.add_argument("--skip-vlm", action="store_true", help="Run YOLO only and skip VLM explanation.")
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
    client = VlmClient(
        model_name=args.vlm_model,
        host=args.ollama_host,
        temperature=0.1,
        num_ctx=args.vlm_num_ctx,
        num_predict=args.vlm_num_predict,
    )
    return VlmService(client=client, prompt_builder=PromptBuilder(), response_parser=VlmResponseParser())


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
            f"box=({detection.x1}, {detection.y1}, {detection.x2}, {detection.y2})"
        )


def main() -> int:
    """Run YOLO and optionally VLM from the terminal."""
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

        inspection_service = InspectionService(
            yolo_service=yolo_service,
            vlm_service=build_vlm_service(args),
        )
        result = inspection_service.inspect(image_path)
        print(f"[INFO] Judgment: {result.status}")
        print(f"[INFO] Detection count: {result.defect_count}")
        print(f"[INFO] Result image: {result.result_image_path}")
        if result.defect_count == 0:
            print("[INFO] VLM analysis skipped")
        print_detection_rows(result.detections)
        if result.vlm_explanation:
            print()
            print("[VLM explanation]")
            print(result.vlm_explanation)

        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
