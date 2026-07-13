from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2

from config.settings import RESULT_IMAGE_DIR
from model.defect_info import Detection
from model.yolo_result import YoloResult
from yolo.model_loader import YoloModelLoader
from yolo.yolo_config import YoloConfig


class YoloDetector:
    def __init__(
        self,
        model_loader: YoloModelLoader | None = None,
        config: YoloConfig | None = None,
    ) -> None:
        self.config = config or YoloConfig()
        self.model_loader = model_loader or YoloModelLoader(self.config)

    def detect(self, image_path: str | Path, output_path: str | Path | None = None) -> YoloResult:
        """Run YOLO inference and save one annotated result image."""
        source_path = Path(image_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Input image not found: {source_path}")

        model = self.model_loader.load()
        prediction = model.predict(
            source=str(source_path),
            imgsz=self.config.image_size,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            save=False,
            verbose=False,
        )
        if not prediction:
            raise RuntimeError(f"YOLO did not return a prediction result for: {source_path}")

        result = prediction[0]
        detections = self._to_detections(result)
        annotated_image = result.plot()
        target_path = Path(output_path) if output_path is not None else self._build_output_path(source_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(target_path), annotated_image):
            raise RuntimeError(f"Failed to save YOLO annotated result image: {target_path}")

        return YoloResult(
            image_path=source_path,
            detections=detections,
            annotated_image_path=target_path,
        )

    def _build_output_path(self, image_path: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return RESULT_IMAGE_DIR / f"{image_path.stem}_yolo_{timestamp}_{uuid4().hex[:8]}{image_path.suffix}"

    def _to_detections(self, result: object) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", {}) or {}
        detections: list[Detection] = []
        xyxy_values = boxes.xyxy.cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist()
        class_values = boxes.cls.cpu().tolist()

        for xyxy, confidence, class_id_float in zip(xyxy_values, confidence_values, class_values, strict=True):
            class_id = int(class_id_float)
            x1, y1, x2, y2 = (int(round(value)) for value in xyxy)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(names.get(class_id, class_id)),
                    confidence=float(confidence),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections
