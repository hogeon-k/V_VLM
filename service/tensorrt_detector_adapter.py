from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from config.settings import RESULT_IMAGE_DIR
from model.defect_info import Detection
from model.yolo_result import YoloResult
from service.detection_location import calculate_detection_location
from service.tensorrt_persistent_worker import (
    TensorRtPersistentWorker,
    TensorRtWorkerError,
    TensorRtWorkerRemoteError,
)

logger = logging.getLogger(__name__)


DEFAULT_TENSORRT_CLASS_NAMES = {
    0: "open_circuit",
    1: "short",
    2: "missing_hole",
}


class TensorRtAdapterError(RuntimeError):
    """Base error for TensorRT adapter configuration and execution failures."""


class TensorRtExecutionError(TensorRtAdapterError):
    """Raised when the TensorRT subprocess fails or times out."""


class TensorRtResultParseError(TensorRtAdapterError):
    """Raised when TensorRT output files are missing or invalid."""


@dataclass(frozen=True, slots=True)
class TensorRtRunMetadata:
    backend: str
    engine_label: str
    engine_path: Path
    image_path: Path
    output_dir: Path
    result_json_path: Path
    detection_count: int
    duration_seconds: float
    returncode: int
    timing_ms: dict[str, float]
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""
    execution_mode: str = "oneshot"
    startup_ms: float | None = None
    ipc_roundtrip_ms: float | None = None


def _resolve_existing_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise TensorRtAdapterError(f"{label} not found: {resolved}")
    return resolved


def _excerpt(value: str, limit: int = 1200) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "... [truncated]"


class TensorRtDetectorAdapter:
    """Run the verified C++ TensorRT CLI and convert its JSON into YoloResult."""

    def __init__(
        self,
        executable_path: str | Path,
        engine_path: str | Path,
        metadata_path: str | Path,
        device_id: int = 0,
        image_size: int = 960,
        confidence_threshold: float = 0.15,
        iou_threshold: float = 0.7,
        engine_label: str = "fp16",
        timeout_seconds: float = 120.0,
        keep_tensorrt_outputs: bool = False,
        class_names: dict[int, str] | None = None,
        use_persistent_worker: bool = True,
        fallback_to_oneshot: bool = True,
        worker_startup_timeout_seconds: float | None = None,
        generate_annotated_image: bool = True,
    ) -> None:
        self.executable_path = _resolve_existing_file(executable_path, "TensorRT executable")
        if self.executable_path.suffix.lower() != ".exe":
            raise TensorRtAdapterError(f"TensorRT executable must be a .exe file: {self.executable_path}")
        self.engine_path = _resolve_existing_file(engine_path, "TensorRT engine")
        if self.engine_path.suffix.lower() not in {".engine", ".plan"}:
            raise TensorRtAdapterError(f"TensorRT engine must be a .engine or .plan file: {self.engine_path}")
        self.metadata_path = _resolve_existing_file(metadata_path, "TensorRT metadata")
        self.device_id = int(device_id)
        self.image_size = int(image_size)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.engine_label = engine_label
        self.timeout_seconds = float(timeout_seconds)
        self.keep_tensorrt_outputs = keep_tensorrt_outputs
        self.class_names = class_names or DEFAULT_TENSORRT_CLASS_NAMES
        self.use_persistent_worker = bool(use_persistent_worker)
        self.fallback_to_oneshot = bool(fallback_to_oneshot)
        self.worker_startup_timeout_seconds = float(
            worker_startup_timeout_seconds
            if worker_startup_timeout_seconds is not None
            else timeout_seconds
        )
        self.generate_annotated_image = bool(generate_annotated_image)
        self.last_metadata: TensorRtRunMetadata | None = None
        self._persistent_worker: TensorRtPersistentWorker | None = None

    def detect(self, image_path: str | Path, output_path: str | Path | None = None) -> YoloResult:
        return self.infer(image_path, output_path=output_path)

    def infer(self, image_path: str | Path, output_path: str | Path | None = None) -> YoloResult:
        source_path = _resolve_existing_file(image_path, "Input image")
        if self.use_persistent_worker:
            try:
                return self._infer_persistent(source_path, output_path)
            except TensorRtWorkerRemoteError as exc:
                raise TensorRtExecutionError(
                    f"TensorRT worker rejected inference: {exc}"
                ) from exc
            except TensorRtWorkerError as exc:
                if not self.fallback_to_oneshot:
                    raise TensorRtExecutionError(
                        f"TensorRT persistent worker failed: {exc}"
                    ) from exc
                logger.warning(
                    "TensorRT persistent worker failed; falling back to one-shot. "
                    "The timed-out request may have completed before termination. error=%s",
                    exc,
                )
        return self._infer_oneshot(source_path, output_path)

    def _infer_persistent(
        self,
        source_path: Path,
        output_path: str | Path | None,
    ) -> YoloResult:
        start = time.perf_counter()
        if self.keep_tensorrt_outputs:
            output_dir = Path(tempfile.mkdtemp(prefix="vvlm_tensorrt_"))
            cleanup_context = None
        else:
            cleanup_context = tempfile.TemporaryDirectory(prefix="vvlm_tensorrt_")
            output_dir = Path(cleanup_context.name)

        try:
            worker = self._get_persistent_worker()
            payload = worker.infer(
                source_path,
                confidence=self.confidence_threshold,
                iou=self.iou_threshold,
                output_dir=output_dir if self.generate_annotated_image else None,
            )
            detections = self._parse_detections(payload, source_path)
            annotated_image_path = (
                self._resolve_annotated_image(output_dir, output_path)
                if self.generate_annotated_image
                else None
            )
            duration_seconds = time.perf_counter() - start
            timing_ms = self._parse_timing_ms(payload)
            ipc_roundtrip_ms = float(payload["ipc_roundtrip_ms"])
            timing_ms["ipc_roundtrip"] = ipc_roundtrip_ms
            self.last_metadata = TensorRtRunMetadata(
                backend=str(payload.get("backend") or "tensorrt"),
                engine_label=str(payload.get("engine_label") or self.engine_label),
                engine_path=self.engine_path,
                image_path=source_path,
                output_dir=output_dir,
                result_json_path=output_dir / "result.json",
                detection_count=len(detections),
                duration_seconds=duration_seconds,
                returncode=0,
                timing_ms=timing_ms,
                stderr_excerpt=worker.stderr_excerpt(),
                execution_mode="persistent",
                startup_ms=worker.startup_ms,
                ipc_roundtrip_ms=ipc_roundtrip_ms,
            )
            logger.info(
                "TensorRT persistent detection completed pid=%s engine_label=%s "
                "image=%s detections=%s duration=%.3fs",
                worker.pid,
                self.engine_label,
                source_path,
                len(detections),
                duration_seconds,
            )
            return YoloResult(
                image_path=source_path,
                detections=detections,
                annotated_image_path=annotated_image_path,
            )
        finally:
            if cleanup_context is not None:
                cleanup_context.cleanup()

    def _infer_oneshot(
        self,
        source_path: Path,
        output_path: str | Path | None,
    ) -> YoloResult:
        start = time.perf_counter()
        if self.keep_tensorrt_outputs:
            output_dir = Path(tempfile.mkdtemp(prefix="vvlm_tensorrt_"))
            cleanup_context = None
        else:
            cleanup_context = tempfile.TemporaryDirectory(prefix="vvlm_tensorrt_")
            output_dir = Path(cleanup_context.name)

        try:
            command = self._build_command(source_path, output_dir)
            completed = self._run_command(command)
            result_json_path = output_dir / "result.json"
            if not result_json_path.is_file():
                raise TensorRtResultParseError(
                    "TensorRT result JSON not found. "
                    f"path={result_json_path}; command={command}; returncode={completed.returncode}; "
                    f"stdout={_excerpt(completed.stdout)}; stderr={_excerpt(completed.stderr)}"
                )

            payload = self._load_result_json(result_json_path)
            detections = self._parse_detections(payload, source_path)
            annotated_image_path = self._resolve_annotated_image(output_dir, output_path)
            duration_seconds = time.perf_counter() - start
            timing_ms = self._parse_timing_ms(payload)
            self.last_metadata = TensorRtRunMetadata(
                backend=str(payload.get("backend") or "tensorrt"),
                engine_label=str(payload.get("engine_label") or self.engine_label),
                engine_path=self.engine_path,
                image_path=source_path,
                output_dir=output_dir,
                result_json_path=result_json_path,
                detection_count=len(detections),
                duration_seconds=duration_seconds,
                returncode=completed.returncode,
                timing_ms=timing_ms,
                stdout_excerpt=_excerpt(completed.stdout),
                stderr_excerpt=_excerpt(completed.stderr),
            )
            logger.info(
                "TensorRT detection completed backend=tensorrt engine_label=%s engine=%s image=%s detections=%s duration=%.3fs returncode=%s result=%s",
                self.engine_label,
                self.engine_path,
                source_path,
                len(detections),
                duration_seconds,
                completed.returncode,
                result_json_path,
            )
            logger.debug("TensorRT stdout: %s", completed.stdout)
            logger.debug("TensorRT stderr: %s", completed.stderr)
            return YoloResult(
                image_path=source_path,
                detections=detections,
                annotated_image_path=annotated_image_path,
            )
        finally:
            if cleanup_context is not None:
                cleanup_context.cleanup()

    def _get_persistent_worker(self) -> TensorRtPersistentWorker:
        if self._persistent_worker is None:
            self._persistent_worker = TensorRtPersistentWorker(
                executable_path=self.executable_path,
                engine_path=self.engine_path,
                metadata_path=self.metadata_path,
                engine_label=self.engine_label,
                device_id=self.device_id,
                image_size=self.image_size,
                startup_timeout_seconds=self.worker_startup_timeout_seconds,
                inference_timeout_seconds=self.timeout_seconds,
            )
        return self._persistent_worker

    def close(self) -> None:
        worker = self._persistent_worker
        self._persistent_worker = None
        if worker is not None:
            worker.stop()

    shutdown = close

    def _build_command(self, image_path: Path, output_dir: Path) -> list[str]:
        return [
            str(self.executable_path),
            "--backend",
            "tensorrt",
            "--engine",
            str(self.engine_path),
            "--metadata",
            str(self.metadata_path),
            "--image",
            str(image_path),
            "--engine-label",
            self.engine_label,
            "--device-id",
            str(self.device_id),
            "--imgsz",
            str(self.image_size),
            "--conf",
            str(self.confidence_threshold),
            "--iou",
            str(self.iou_threshold),
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--output",
            str(output_dir),
        ]

    def _run_command(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            completed_bytes = subprocess.run(
                command,
                shell=False,
                capture_output=True,
                text=False,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = self._decode_process_output(exc.stdout)
            stderr = self._decode_process_output(exc.stderr)
            raise TensorRtExecutionError(
                "TensorRT subprocess timed out. "
                f"timeout={self.timeout_seconds}; executable={self.executable_path}; engine={self.engine_path}; "
                f"command={command}; "
                f"stdout={_excerpt(stdout)}; stderr={_excerpt(stderr)}"
            ) from exc

        completed = subprocess.CompletedProcess(
            args=completed_bytes.args,
            returncode=completed_bytes.returncode,
            stdout=self._decode_process_output(completed_bytes.stdout),
            stderr=self._decode_process_output(completed_bytes.stderr),
        )
        if completed.returncode != 0:
            raise TensorRtExecutionError(
                "TensorRT subprocess failed. "
                f"returncode={completed.returncode}; executable={self.executable_path}; engine={self.engine_path}; "
                f"command={command}; "
                f"stdout={_excerpt(completed.stdout)}; stderr={_excerpt(completed.stderr)}"
            )
        return completed

    @staticmethod
    def _decode_process_output(data: bytes | str | None) -> str:
        if not data:
            return ""
        if isinstance(data, str):
            return data
        for encoding in ("utf-8", "cp949"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    def _load_result_json(self, result_json_path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(
                result_json_path.read_text(encoding="utf-8-sig"),
                parse_constant=self._reject_json_constant,
            )
        except UnicodeDecodeError as exc:
            raise TensorRtResultParseError(f"TensorRT result JSON is not valid UTF-8: {result_json_path}; {exc}") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise TensorRtResultParseError(f"TensorRT result JSON is invalid: {result_json_path}; {exc}") from exc
        if not isinstance(payload, dict):
            raise TensorRtResultParseError(f"TensorRT result JSON root must be an object: {result_json_path}")
        return payload

    def _parse_detections(self, payload: dict[str, Any], source_path: Path) -> list[Detection]:
        raw_detections = payload.get("detections")
        if not isinstance(raw_detections, list):
            raise TensorRtResultParseError("TensorRT result JSON field 'detections' must be a list.")

        image_width, image_height = self._read_image_size(source_path)
        detections: list[Detection] = []
        for index, raw in enumerate(raw_detections):
            if not isinstance(raw, dict):
                raise TensorRtResultParseError(f"TensorRT detection #{index} must be an object.")
            class_id = self._parse_class_id(raw.get("class_id"), index)
            class_name = str(raw.get("class_name") or self.class_names.get(class_id, class_id))
            confidence = self._parse_confidence(raw.get("confidence"), index)
            x1, y1, x2, y2 = self._parse_bbox(raw.get("bbox"), index)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    location=calculate_detection_location(
                        (x1, y1, x2, y2),
                        image_width=image_width,
                        image_height=image_height,
                    ),
                )
            )
        return detections

    def _parse_class_id(self, value: object, index: int) -> int:
        if type(value) is not int:
            raise TensorRtResultParseError(f"TensorRT detection #{index} class_id must be an integer.")
        if value not in self.class_names:
            raise TensorRtResultParseError(f"TensorRT detection #{index} class_id out of range: {value}")
        return value

    def _parse_confidence(self, value: object, index: int) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TensorRtResultParseError(f"TensorRT detection #{index} confidence must be numeric.")
        confidence = float(value)
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise TensorRtResultParseError(f"TensorRT detection #{index} confidence out of range: {confidence}")
        return confidence

    def _parse_bbox(self, value: object, index: int) -> tuple[int, int, int, int]:
        if not isinstance(value, list) or len(value) != 4:
            raise TensorRtResultParseError(f"TensorRT detection #{index} bbox must be [x1, y1, x2, y2].")
        if not all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        ):
            raise TensorRtResultParseError(f"TensorRT detection #{index} bbox values must be numeric.")
        x1, y1, x2, y2 = (int(round(float(item))) for item in value)
        if x2 < x1 or y2 < y1:
            raise TensorRtResultParseError(f"TensorRT detection #{index} bbox has invalid order: {value}")
        return x1, y1, x2, y2

    def _read_image_size(self, source_path: Path) -> tuple[int, int]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise TensorRtResultParseError("Pillow is required to read image size for TensorRT detections.") from exc

        with Image.open(source_path) as image:
            return int(image.width), int(image.height)

    def _parse_timing_ms(self, payload: dict[str, Any]) -> dict[str, float]:
        timing = payload.get("timing_ms")
        if not isinstance(timing, dict):
            return {}
        parsed: dict[str, float] = {}
        for key, value in timing.items():
            if (
                isinstance(key, str)
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            ):
                parsed[key] = float(value)
        return parsed

    @staticmethod
    def _reject_json_constant(value: str) -> object:
        raise ValueError(f"non-standard numeric constant {value}")

    def _resolve_annotated_image(self, output_dir: Path, output_path: str | Path | None) -> Path | None:
        result_image = output_dir / "result.jpg"
        if not result_image.is_file():
            return None
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            target = RESULT_IMAGE_DIR / f"tensorrt_{timestamp}_{uuid4().hex[:8]}.jpg"
        else:
            target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_image, target)
        return target
