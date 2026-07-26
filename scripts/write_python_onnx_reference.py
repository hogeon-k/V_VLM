from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.onnx_detector import OnnxDetector, class_aware_nms, preprocess_image, restore_boxes_to_original, validate_onnx_output, xywh_to_xyxy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Python ONNX Runtime single-image reference result.")
    parser.add_argument("--model", type=Path, default=Path("models/best.onnx"))
    parser.add_argument("--metadata", type=Path, default=Path("models/model_metadata.json"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/cpp_onnx/reference/python_onnx_result.json"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    return parser.parse_args(argv)


def load_class_names(path: Path) -> dict[int, str]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        names = data.get("class_names")
        if isinstance(names, list) and names:
            return {index: str(name) for index, name in enumerate(names)}
    return {0: "open_circuit", 1: "short", 2: "missing_hole"}


def postprocess_float(output: np.ndarray, letterbox_info: Any, conf: float, iou: float, class_names: dict[int, str]) -> list[dict[str, Any]]:
    predictions = validate_onnx_output(output)
    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
    confidences = np.max(class_scores, axis=1).astype(np.float32)
    candidates = confidences >= conf
    if not np.any(candidates):
        return []

    boxes = xywh_to_xyxy(boxes_xywh[candidates])
    boxes = restore_boxes_to_original(boxes, letterbox_info)
    scores = confidences[candidates]
    classes = class_ids[candidates]
    keep = class_aware_nms(boxes, scores, classes, iou)

    detections: list[dict[str, Any]] = []
    for index in keep:
        class_id = int(classes[index])
        x1, y1, x2, y2 = (float(value) for value in boxes[index])
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "confidence": float(scores[index]),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return detections


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    class_names = load_class_names(args.metadata)
    detector = OnnxDetector(
        args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        class_names=class_names,
        requested_provider=args.provider,
        preload_torch_cuda=False,
    )
    session = detector._load_session()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Input image not found or unreadable: {args.image}")

    total_start = time.perf_counter()
    start = time.perf_counter()
    tensor, letterbox_info = preprocess_image(image, args.imgsz)
    preprocess_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    outputs = session.run([detector._output_name], {detector._input_name: tensor})
    inference_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    output = np.asarray(outputs[0])
    detections = postprocess_float(output, letterbox_info, args.conf, args.iou, class_names)
    postprocess_ms = (time.perf_counter() - start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000
    data = {
        "model": str(args.model),
        "image": str(args.image),
        "provider": detector.actual_providers[0] if detector.actual_providers else "",
        "input_name": detector._input_name,
        "output_name": detector._output_name,
        "input_shape": detector._input_shape,
        "output_shape": list(output.shape),
        "config": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou},
        "timing_ms": {
            "preprocess": preprocess_ms,
            "inference": inference_ms,
            "postprocess": postprocess_ms,
            "total": total_ms,
        },
        "detections": detections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote Python ONNX reference: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
