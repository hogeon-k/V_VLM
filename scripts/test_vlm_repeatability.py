from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_yolo_vlm import build_vlm_service, build_yolo_service, positive_int


def parse_args() -> argparse.Namespace:
    """Parse repeatability test options."""
    parser = argparse.ArgumentParser(
        description="Run repeated VLM requests for the same YOLO result and compare outputs."
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
    parser.add_argument("--save-crop-montage", action="store_true", help="Save generated crop montage images.")
    parser.add_argument(
        "--crop-montage-output-dir",
        default="data/result_images/montage",
        help="Directory used when --save-crop-montage is enabled.",
    )
    parser.add_argument("--repeat-count", type=int, default=5, help="Number of repeated VLM runs.")
    return parser.parse_args()


def canonical_parsed_json(parse_result: object) -> str:
    """Return canonical JSON for parsed VLM data or an empty string on fallback."""
    parsed_response = getattr(parse_result, "parsed_response", None)
    if parsed_response is None:
        return ""
    return json.dumps(
        parsed_response.raw_data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compare_values(values: list[str]) -> tuple[int, bool]:
    """Return exact match count against the first value and whether all values match."""
    if not values:
        return 0, True
    first = values[0]
    matches = sum(1 for value in values if value == first)
    return matches, matches == len(values)


def exact_match_label(values: list[str]) -> str:
    """Return true/false for 2+ comparable values, otherwise N/A."""
    if len(values) < 2:
        return "N/A"
    _, all_match = compare_values(values)
    return str(all_match).lower()


def failure_signature(vlm_service: object) -> str:
    """Return a stable failure signature without volatile paths or timings."""
    metadata = getattr(vlm_service, "last_ollama_metadata", None)
    return json.dumps(
        {
            "vlm_status": getattr(vlm_service, "last_vlm_status", "unknown"),
            "parse_status": getattr(vlm_service, "last_parse_status", "unknown"),
            "fallback_used": getattr(vlm_service, "last_fallback_used", False),
            "ollama_done": getattr(metadata, "done", None),
            "ollama_content_length": getattr(metadata, "content_length", None),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> int:
    """Run YOLO once and repeat the VLM call with identical settings."""
    args = parse_args()
    if args.repeat_count < 1:
        print("[ERROR] --repeat-count must be at least 1", file=sys.stderr)
        return 1

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    yolo_result = build_yolo_service(args).detect(image_path)
    if yolo_result.defect_count == 0:
        print("[INFO] YOLO detections: 0")
        print("[INFO] VLM repeatability skipped because no defect was detected.")
        return 0

    raw_responses: list[str] = []
    canonical_json_values: list[str] = []
    successful_raw_responses: list[str] = []
    successful_canonical_json_values: list[str] = []
    failure_signatures: list[str] = []
    final_judgments: list[str] = []
    detection_counts: list[int] = []
    detection_ids: list[list[int]] = []
    parse_success_count = 0
    fallback_count = 0
    empty_response_count = 0
    quality_acceptable_count = 0
    quality_warning_count = 0
    total_class_name_only_count = 0
    runs_with_class_name_only = 0
    summary_contradiction_count = 0
    runs_with_summary_contradiction = 0

    vlm_image_path = yolo_result.annotated_image_path or image_path
    print(f"[INFO] VLM full image size limit: {args.vlm_full_image_size or args.vlm_image_size}")
    print(f"[INFO] VLM montage size limit: {args.vlm_montage_size or args.vlm_crop_montage_size}")
    print(f"[INFO] VLM image mode: {args.vlm_image_mode}")
    for index in range(1, args.repeat_count + 1):
        print(f"[INFO] Repeat run {index}/{args.repeat_count}")
        vlm_service = build_vlm_service(args)
        vlm_service.describe_defects(vlm_image_path, yolo_result)
        if index == 1 and vlm_service.last_preparation_info is not None:
            info = vlm_service.last_preparation_info
            print(f"[INFO] VLM image mode: {info.image_mode}")
            print(f"[INFO] VLM image count: {info.image_count}")
            if info.full_image_size is not None:
                print(f"[INFO] VLM full image size: {info.full_image_size[0]}x{info.full_image_size[1]}")
            if info.crop_montage_size is not None:
                print(f"[INFO] VLM crop montage size: {info.crop_montage_size[0]}x{info.crop_montage_size[1]}")
        parse_result = vlm_service.last_parse_result
        quality = vlm_service.last_quality_info
        if quality.quality_status == "acceptable":
            quality_acceptable_count += 1
        elif quality.quality_status == "warning":
            quality_warning_count += 1
        total_class_name_only_count += quality.class_name_only_count
        if quality.class_name_only_count > 0:
            runs_with_class_name_only += 1
        if quality.summary_contradiction:
            summary_contradiction_count += 1
            runs_with_summary_contradiction += 1
        raw_response = vlm_service.last_raw_response or ""
        raw_responses.append(raw_response)
        if not raw_response:
            empty_response_count += 1
        canonical_json = canonical_parsed_json(parse_result)
        canonical_json_values.append(canonical_json)
        if parse_result is not None and parse_result.parse_success and parse_result.parsed_response:
            parse_success_count += 1
            successful_raw_responses.append(raw_response)
            successful_canonical_json_values.append(canonical_json)
            parsed = parse_result.parsed_response
            final_judgments.append(parsed.final_judgment)
            detection_counts.append(len(parsed.detections))
            detection_ids.append([detection.detection_id for detection in parsed.detections])
        else:
            fallback_count += 1
            failure_signatures.append(failure_signature(vlm_service))

    raw_match_count, raw_all_match = compare_values(raw_responses)
    parsed_match_count, parsed_all_match = compare_values(canonical_json_values)
    _, final_judgment_consistent = compare_values(final_judgments)
    _, detection_count_consistent = compare_values([str(value) for value in detection_counts])
    _, detection_id_consistent = compare_values(
        [json.dumps(value, separators=(",", ":")) for value in detection_ids]
    )

    print("[INFO] Repeatability summary")
    print(f"[INFO] Total runs: {args.repeat_count}")
    print(f"[INFO] Run count: {args.repeat_count}")
    print(f"[INFO] Successful response count: {len(successful_raw_responses)}")
    print(f"[INFO] Empty response count: {empty_response_count}")
    print(f"[INFO] Parse success count: {parse_success_count}")
    print(f"[INFO] Parse failure count: {args.repeat_count - parse_success_count}")
    print(f"[INFO] Fallback count: {fallback_count}")
    print(f"[INFO] Quality acceptable count: {quality_acceptable_count}")
    print(f"[INFO] Quality warning count: {quality_warning_count}")
    print(f"[INFO] Total class-name-only count: {total_class_name_only_count}")
    print(f"[INFO] Runs with class-name-only: {runs_with_class_name_only}")
    print(f"[INFO] Summary contradiction count: {summary_contradiction_count}")
    print(f"[INFO] Runs with summary contradiction: {runs_with_summary_contradiction}")
    print(
        "[INFO] Class-name-only run rate: "
        f"{runs_with_class_name_only / args.repeat_count * 100:.1f}%"
    )
    print(
        "[INFO] Summary contradiction run rate: "
        f"{runs_with_summary_contradiction / args.repeat_count * 100:.1f}%"
    )
    print(f"[INFO] Raw response exact match count: {raw_match_count}/{args.repeat_count}")
    print(f"[INFO] Parsed response exact match count: {parsed_match_count}/{args.repeat_count}")
    print(f"[INFO] Final judgment consistency: {str(final_judgment_consistent).lower()}")
    print(f"[INFO] Detection count consistency: {str(detection_count_consistent).lower()}")
    print(f"[INFO] Detection ID consistency: {str(detection_id_consistent).lower()}")
    print("[INFO] Raw response SHA-256 values:")
    for value in raw_responses:
        print(f"[INFO] - {sha256_text(value)}")
    print("[INFO] Normalized parsed JSON SHA-256 values:")
    for value in canonical_json_values:
        print(f"[INFO] - {sha256_text(value)}")
    print(f"[INFO] Legacy raw exact match: {str(raw_all_match).lower()}")
    print(
        "[INFO] Successful non-empty responses all match: "
        f"{exact_match_label(successful_raw_responses)}"
    )
    print(f"[INFO] Failure patterns all match: {exact_match_label(failure_signatures)}")
    print(
        "[INFO] Parsed successful results all match: "
        f"{exact_match_label(successful_canonical_json_values)}"
    )
    print(f"[INFO] Legacy parsed responses all match: {str(parsed_all_match).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
