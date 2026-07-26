from __future__ import annotations

import importlib.util
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from model.defect_info import Detection
from model.yolo_result import YoloResult


DEFAULT_CLASS_NAMES = {
    0: "open_circuit",
    1: "short",
    2: "missing_hole",
}

_DLL_DIRECTORY_HANDLES: dict[Path, object | None] = {}


@dataclass(frozen=True, slots=True)
class TorchCudaPreloadResult:
    attempted: bool
    success: bool
    error: str | None = None
    cuda_available: bool | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    cudnn_version: int | None = None


def package_subdir(package_name: str, *parts: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(package_name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    locations = spec.submodule_search_locations
    if locations:
        base = Path(next(iter(locations)))
    elif spec.origin:
        base = Path(spec.origin).resolve().parent
    else:
        return None
    return base.joinpath(*parts)


def windows_dll_directory_candidates() -> list[Path]:
    candidates: list[Path] = [
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
        Path(sys.prefix) / "Lib" / "site-packages" / "onnxruntime" / "capi",
    ]
    torch_lib = package_subdir("torch", "lib")
    ort_capi = package_subdir("onnxruntime", "capi")
    for candidate in (torch_lib, ort_capi):
        if candidate is not None:
            candidates.append(candidate)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def register_windows_dll_directories() -> list[Path]:
    if os.name != "nt":
        return []

    registered: list[Path] = []
    for candidate in windows_dll_directory_candidates():
        if not candidate.is_dir():
            continue
        if candidate in registered:
            continue
        if candidate not in _DLL_DIRECTORY_HANDLES:
            handle = os.add_dll_directory(str(candidate)) if hasattr(os, "add_dll_directory") else None
            _DLL_DIRECTORY_HANDLES[candidate] = handle
        registered.append(candidate)
    return registered


def preload_torch_cuda_dlls() -> TorchCudaPreloadResult:
    try:
        import torch
    except Exception as exc:
        return TorchCudaPreloadResult(attempted=True, success=False, error=f"{type(exc).__name__}: {exc}")

    try:
        cuda_available = bool(torch.cuda.is_available())
        result = {
            "cuda_available": cuda_available,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if not cuda_available:
            return TorchCudaPreloadResult(attempted=True, success=False, error="torch.cuda.is_available() returned False", **result)
        torch.cuda.init()
        tensor = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        del tensor
        return TorchCudaPreloadResult(attempted=True, success=True, **result)
    except Exception as exc:
        return TorchCudaPreloadResult(attempted=True, success=False, error=f"{type(exc).__name__}: {exc}")


@dataclass(frozen=True, slots=True)
class LetterboxInfo:
    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    ratio: tuple[float, float]
    pad: tuple[int, int]
    new_unpad: tuple[int, int]


@dataclass(frozen=True, slots=True)
class TimedDetections:
    detections: list[Detection]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    providers: list[str]
    input_name: str
    output_name: str
    input_shape: list[Any]
    output_shape: list[int]


def letterbox(
    image: np.ndarray,
    new_shape: int | tuple[int, int] = 960,
    padding_value: int = 114,
    stride: int = 32,
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    center: bool = True,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize and pad like Ultralytics LetterBox for fixed-size detection input."""
    if isinstance(new_shape, int):
        target_shape = (new_shape, new_shape)
    else:
        target_shape = new_shape

    shape = image.shape[:2]
    ratio = min(target_shape[0] / shape[0], target_shape[1] / shape[1])
    if not scaleup:
        ratio = min(ratio, 1.0)

    new_unpad = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = target_shape[1] - new_unpad[0]
    dh = target_shape[0] - new_unpad[1]

    if auto:
        dw = int(np.mod(dw, stride))
        dh = int(np.mod(dh, stride))
    elif scale_fill:
        dw = 0
        dh = 0
        new_unpad = (target_shape[1], target_shape[0])
        ratio_pair = (target_shape[1] / shape[1], target_shape[0] / shape[0])
    else:
        ratio_pair = (ratio, ratio)

    if not scale_fill:
        ratio_pair = (ratio, ratio)

    if center:
        dw /= 2
        dh /= 2

    top = round(dh - 0.1) if center else 0
    bottom = round(dh + 0.1)
    left = round(dw - 0.1) if center else 0
    right = round(dw + 0.1)

    resized = image
    if shape[::-1] != new_unpad:
        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(padding_value, padding_value, padding_value),
    )
    info = LetterboxInfo(
        original_shape=(int(shape[0]), int(shape[1])),
        resized_shape=(int(padded.shape[0]), int(padded.shape[1])),
        ratio=ratio_pair,
        pad=(int(left), int(top)),
        new_unpad=(int(new_unpad[0]), int(new_unpad[1])),
    )
    return padded, info


def preprocess_image(image: np.ndarray, imgsz: int) -> tuple[np.ndarray, LetterboxInfo]:
    padded, info = letterbox(image, new_shape=imgsz, padding_value=114, auto=False, scaleup=True, center=True)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor), info


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.astype(np.float32, copy=True)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def restore_boxes_to_original(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    restored = boxes.astype(np.float32, copy=True)
    pad_x, pad_y = info.pad
    gain_x, gain_y = info.ratio
    restored[:, [0, 2]] -= pad_x
    restored[:, [1, 3]] -= pad_y
    restored[:, [0, 2]] /= gain_x
    restored[:, [1, 3]] /= gain_y
    return clip_boxes(restored, width=info.original_shape[1], height=info.original_shape[0])


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = boxes.astype(np.float32, copy=True)
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, width)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, height)
    return clipped


def bbox_iou(box_a: list[float] | np.ndarray, box_b: list[float] | np.ndarray) -> float:
    a = np.asarray(box_a, dtype=np.float32)
    b = np.asarray(box_b, dtype=np.float32)
    inter_x1 = max(float(a[0]), float(b[0]))
    inter_y1 = max(float(a[1]), float(b[1]))
    inter_x2 = min(float(a[2]), float(b[2]))
    inter_y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    keep: list[int] = []
    for class_id in sorted(set(int(value) for value in class_ids.tolist())):
        indices = np.where(class_ids == class_id)[0]
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size > 0:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            ious = np.array([bbox_iou(boxes[current], boxes[int(index)]) for index in remaining], dtype=np.float32)
            order = remaining[ious <= iou_threshold]
    return sorted(keep, key=lambda index: float(scores[index]), reverse=True)


def validate_onnx_output(output: np.ndarray) -> np.ndarray:
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"Expected ONNX output shape [1, 7, N], got {list(output.shape)}")
    if output.shape[1] < 5:
        raise ValueError(f"Expected at least 5 channels in ONNX output, got {list(output.shape)}")
    return np.transpose(output[0], (1, 0))


def postprocess_output(
    output: np.ndarray,
    letterbox_info: LetterboxInfo,
    conf_threshold: float,
    iou_threshold: float,
    class_names: dict[int, str] | None = None,
) -> list[Detection]:
    predictions = validate_onnx_output(output)
    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
    confidences = np.max(class_scores, axis=1).astype(np.float32)
    candidates = confidences >= conf_threshold
    if not np.any(candidates):
        return []

    boxes = xywh_to_xyxy(boxes_xywh[candidates])
    boxes = restore_boxes_to_original(boxes, letterbox_info)
    scores = confidences[candidates]
    classes = class_ids[candidates]
    keep = class_aware_nms(boxes, scores, classes, iou_threshold)
    names = class_names or DEFAULT_CLASS_NAMES

    detections: list[Detection] = []
    for index in keep:
        x1, y1, x2, y2 = (int(round(float(value))) for value in boxes[index])
        class_id = int(classes[index])
        detections.append(
            Detection(
                class_id=class_id,
                class_name=str(names.get(class_id, class_id)),
                confidence=float(scores[index]),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return detections


def detection_to_dict(detection: Detection) -> dict[str, Any]:
    return {
        "class_id": int(detection.class_id),
        "class_name": str(detection.class_name),
        "confidence": float(detection.confidence),
        "bbox": [int(detection.x1), int(detection.y1), int(detection.x2), int(detection.y2)],
    }


def summarize_timings(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    keys = ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms")
    for key in keys:
        values = [sample[key] for sample in samples]
        summary[key] = {
            "avg": float(statistics.fmean(values)) if values else 0.0,
            "min": float(min(values)) if values else 0.0,
            "max": float(max(values)) if values else 0.0,
            "median": float(statistics.median(values)) if values else 0.0,
            "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        }
    return summary


class OnnxDetector:
    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 960,
        conf: float = 0.15,
        iou: float = 0.5,
        class_names: dict[int, str] | None = None,
        requested_provider: str = "CUDAExecutionProvider",
        require_cuda: bool = False,
        preload_torch_cuda: bool = True,
        register_windows_dlls: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.require_cuda = require_cuda
        self.preload_torch_cuda = preload_torch_cuda
        self.register_windows_dlls = register_windows_dlls
        self.requested_provider = requested_provider
        self.actual_providers: list[str] = []
        self.using_cuda = False
        self.registered_dll_directories: list[str] = []
        self.cuda_preload_attempted = False
        self.cuda_preload_success = False
        self.cuda_initialization_error: str | None = None
        self._session: Any | None = None
        self._input_name = ""
        self._output_name = ""
        self._input_shape: list[Any] = []

    def _requested_providers(self, available: list[str]) -> list[Any]:
        if self.requested_provider == "CPUExecutionProvider":
            return ["CPUExecutionProvider"]
        if self.requested_provider in available:
            return [(self.requested_provider, {"device_id": 0}), "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_session(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")
        if self._session is not None:
            return self._session

        if self.register_windows_dlls:
            self.registered_dll_directories = [str(path) for path in register_windows_dll_directories()]
        if self.requested_provider == "CUDAExecutionProvider" and self.preload_torch_cuda:
            preload = preload_torch_cuda_dlls()
            self.cuda_preload_attempted = preload.attempted
            self.cuda_preload_success = preload.success
            self.cuda_initialization_error = preload.error
            if self.require_cuda and not preload.success:
                raise RuntimeError(
                    "CUDAExecutionProvider was required, but PyTorch CUDA preload failed. "
                    "Run scripts/diagnose_onnxruntime_cuda.py to inspect CUDA, cuDNN, DLL, and PATH dependencies. "
                    f"Reason: {preload.error}"
                )

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed. Install onnxruntime-gpu or onnxruntime.") from exc

        available = ort.get_available_providers()
        providers = self._requested_providers(available)

        try:
            self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        except Exception as exc:
            if self.require_cuda:
                raise RuntimeError(
                    "CUDAExecutionProvider was required, but ONNX Runtime failed to create a CUDA session. "
                    "Run scripts/diagnose_onnxruntime_cuda.py to inspect CUDA, cuDNN, DLL, and PATH dependencies. "
                    f"Reason: {type(exc).__name__}: {exc}"
                ) from exc
            self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])

        self.actual_providers = list(self._session.get_providers())
        self.using_cuda = "CUDAExecutionProvider" in self.actual_providers
        if self.require_cuda and not self.using_cuda:
            raise RuntimeError(
                "CUDAExecutionProvider was required, but ONNX Runtime created a CPU-only session. "
                "Run scripts/diagnose_onnxruntime_cuda.py to inspect CUDA, cuDNN, DLL, and PATH dependencies."
            )

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected one ONNX input, got {len(inputs)}")
        if len(outputs) != 1:
            raise ValueError(f"Expected one ONNX output, got {len(outputs)}")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._input_shape = list(inputs[0].shape)
        self._validate_input_shape(self._input_shape)
        return self._session

    def _validate_input_shape(self, shape: list[Any]) -> None:
        if len(shape) != 4:
            raise ValueError(f"Expected ONNX input rank 4 [1, 3, H, W], got {shape}")
        if shape[1] not in (3, "3"):
            raise ValueError(f"Expected ONNX input channel dimension 3, got {shape}")
        expected = [self.imgsz, self.imgsz]
        actual_hw = shape[2:4]
        concrete = [value for value in actual_hw if isinstance(value, int)]
        if concrete and actual_hw != expected:
            raise ValueError(f"Expected ONNX input size {expected}, got {shape}")

    def detect(self, image_path: str | Path) -> YoloResult:
        return YoloResult(image_path=Path(image_path), detections=self.detect_timed(image_path).detections)

    def detect_timed(self, image_path: str | Path) -> TimedDetections:
        session = self._load_session()
        source_path = Path(image_path)
        image = cv2.imread(str(source_path))
        if image is None:
            raise FileNotFoundError(f"Input image not found or unreadable: {source_path}")

        total_start = time.perf_counter()
        start = time.perf_counter()
        input_tensor, letterbox_info = preprocess_image(image, self.imgsz)
        preprocess_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        outputs = session.run([self._output_name], {self._input_name: input_tensor})
        inference_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        output = np.asarray(outputs[0])
        detections = postprocess_output(output, letterbox_info, self.conf, self.iou, self.class_names)
        postprocess_ms = (time.perf_counter() - start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000

        return TimedDetections(
            detections=detections,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            total_ms=total_ms,
            providers=list(session.get_providers()),
            input_name=self._input_name,
            output_name=self._output_name,
            input_shape=self._input_shape,
            output_shape=list(output.shape),
        )
