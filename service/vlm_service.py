from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from model.yolo_result import YoloResult
from vlm.crop_montage import create_crop_montage_jpeg_bytes
from vlm.image_preprocessor import read_image_size_from_bytes, resize_image_to_jpeg_bytes
from vlm.prompt_builder import PromptBuilder
from vlm.response_parser import VlmResponseParser
from vlm.vlm_client import VlmClient


@dataclass(slots=True)
class VlmPreparationInfo:
    image_count: int = 0
    detection_crop_count: int = 0
    full_image_size: tuple[int, int] | None = None
    crop_montage_size: tuple[int, int] | None = None
    image_preparation_seconds: float | None = None
    inference_seconds: float | None = None


class VlmService:
    def __init__(
        self,
        client: VlmClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: VlmResponseParser | None = None,
        image_size: int = 960,
        image_quality: int = 90,
        crop_montage_size: int = 960,
        crop_padding: int = 192,
        crop_min_size: int = 256,
        crop_max_size: int = 512,
    ) -> None:
        self.client = client or VlmClient()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_parser = response_parser or VlmResponseParser()
        self.image_size = image_size
        self.image_quality = image_quality
        self.crop_montage_size = crop_montage_size
        self.crop_padding = crop_padding
        self.crop_min_size = crop_min_size
        self.crop_max_size = crop_max_size
        self.last_preparation_info: VlmPreparationInfo | None = None

    def describe_defects(self, image_path: Path, yolo_result: YoloResult) -> str | None:
        """Explain NG detections with a local Ollama VLM."""
        if yolo_result.defect_count == 0:
            return None

        prompt = self.prompt_builder.build_defect_prompt(yolo_result)
        preparation_started = perf_counter()
        full_image_bytes = resize_image_to_jpeg_bytes(
            image_path,
            max_size=self.image_size,
            quality=self.image_quality,
        )
        montage_bytes = create_crop_montage_jpeg_bytes(
            image_path=image_path,
            detections=yolo_result.detections,
            max_size=self.crop_montage_size,
            quality=self.image_quality,
            padding=self.crop_padding,
            min_crop_size=self.crop_min_size,
            max_crop_size=self.crop_max_size,
        )
        preparation_seconds = perf_counter() - preparation_started

        self.last_preparation_info = VlmPreparationInfo(
            image_count=2,
            detection_crop_count=yolo_result.defect_count,
            full_image_size=read_image_size_from_bytes(full_image_bytes),
            crop_montage_size=read_image_size_from_bytes(montage_bytes),
            image_preparation_seconds=preparation_seconds,
        )

        inference_started = perf_counter()
        try:
            response = self.client.generate(
                prompt,
                image_bytes_list=[
                    full_image_bytes,
                    montage_bytes,
                ],
            )
        finally:
            self.last_preparation_info.inference_seconds = (
                perf_counter() - inference_started
            )
        return self.response_parser.parse_description(response)
