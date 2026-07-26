from __future__ import annotations

import logging

from config.detector_settings import DetectorSettings, resolve_project_path, validate_tensorrt_settings
from config.settings import YOLO_MODEL_PATH
from service.onnx_detector import OnnxDetector
from service.tensorrt_detector_adapter import TensorRtDetectorAdapter
from service.yolo_service import YoloService
from yolo.detector import YoloDetector
from yolo.model_loader import YoloModelLoader
from yolo.yolo_config import YoloConfig

logger = logging.getLogger(__name__)


def create_yolo_service_from_settings(detector_settings: DetectorSettings) -> YoloService:
    if detector_settings.detector_backend == "tensorrt":
        validate_tensorrt_settings(detector_settings)
        detector = TensorRtDetectorAdapter(
            executable_path=resolve_project_path(detector_settings.tensorrt_executable_path),
            engine_path=resolve_project_path(detector_settings.tensorrt_engine_path),
            metadata_path=resolve_project_path(detector_settings.tensorrt_metadata_path),
            device_id=detector_settings.tensorrt_device_id,
            engine_label=detector_settings.tensorrt_engine_label,
            timeout_seconds=detector_settings.tensorrt_timeout_seconds,
            keep_tensorrt_outputs=detector_settings.keep_tensorrt_outputs,
        )
        return YoloService(detector)

    if detector_settings.detector_backend == "onnx":
        detector = OnnxDetector(resolve_project_path("models/best.onnx"))
        logger.info(
            "ONNX backend selected: implementation=%s model=%s provider=%s",
            detector.__class__.__name__,
            detector.model_path,
            detector.requested_provider,
        )
        return YoloService(detector)

    config = YoloConfig(model_path=YOLO_MODEL_PATH)
    loader = YoloModelLoader(config)
    return YoloService(YoloDetector(model_loader=loader, config=config))
