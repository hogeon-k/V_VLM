from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

from model.yolo_result import YoloResult
from vlm.crop_montage import create_crop_montage_result, save_montage_bytes
from vlm.image_preprocessor import read_image_size_from_bytes, resize_image_to_jpeg_bytes
from vlm.ollama_response import OllamaContentError, OllamaResponseMetadata
from vlm.prompt_builder import PromptBuilder
from vlm.response_parser import (
    VlmQualityInfo,
    VlmParseResult,
    VlmResponseParser,
    format_yolo_fallback_response,
)
from vlm.vlm_client import VlmClient


@dataclass(slots=True)
class VlmPreparationInfo:
    image_mode: str = "full_montage"
    image_count: int = 0
    detection_crop_count: int = 0
    full_image_prepared: bool = False
    crop_montage_prepared: bool = False
    full_image_size_limit: int | None = None
    crop_montage_size_limit: int | None = None
    full_image_size: tuple[int, int] | None = None
    crop_montage_size: tuple[int, int] | None = None
    crop_montage_path: Path | None = None
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
        save_crop_montage: bool = False,
        crop_montage_output_dir: str | Path = "data/result_images/montage",
        image_mode: str = "full_montage",
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
        self.save_crop_montage = save_crop_montage
        self.crop_montage_output_dir = Path(crop_montage_output_dir)
        self.image_mode = image_mode
        self.last_preparation_info: VlmPreparationInfo | None = None
        self.last_raw_response: str | None = None
        self.last_crop_montage_path: Path | None = None
        self.last_parse_result: VlmParseResult | None = None
        self.last_parse_success: bool = False
        self.last_parse_error: str = ""
        self.last_fallback_used: bool = False
        self.last_vlm_status: str = "not_run"
        self.last_parse_status: str = "not_attempted"
        self.last_ollama_metadata: OllamaResponseMetadata | None = None
        self.last_quality_info: VlmQualityInfo = VlmQualityInfo()

    def describe_defects(self, image_path: Path, yolo_result: YoloResult) -> str | None:
        """Explain NG detections with a local Ollama VLM."""
        self.last_preparation_info = None
        self.last_raw_response = None
        self.last_crop_montage_path = None
        self.last_parse_result = None
        self.last_parse_success = False
        self.last_parse_error = ""
        self.last_fallback_used = False
        self.last_vlm_status = "not_run"
        self.last_parse_status = "not_attempted"
        self.last_ollama_metadata = None
        self.last_quality_info = VlmQualityInfo()
        if yolo_result.defect_count == 0:
            return None

        prompt = self.prompt_builder.build_defect_prompt(yolo_result, image_mode=self.image_mode)
        preparation_started = perf_counter()
        full_image_bytes = resize_image_to_jpeg_bytes(
            image_path,
            max_size=self.image_size,
            quality=self.image_quality,
        )
        montage_result = create_crop_montage_result(
            image_path=image_path,
            detections=yolo_result.detections,
            max_size=self.crop_montage_size,
            quality=self.image_quality,
            padding=self.crop_padding,
            min_crop_size=self.crop_min_size,
            max_crop_size=self.crop_max_size,
        )
        montage_bytes = montage_result.image_bytes
        crop_montage_path = self._save_crop_montage_if_enabled(
            montage_bytes=montage_bytes,
            source_image_path=yolo_result.image_path,
        )
        preparation_seconds = perf_counter() - preparation_started
        image_bytes_list = self._image_bytes_for_mode(
            full_image_bytes=full_image_bytes,
            montage_bytes=montage_bytes,
        )

        self.last_preparation_info = VlmPreparationInfo(
            image_mode=self.image_mode,
            image_count=len(image_bytes_list),
            detection_crop_count=montage_result.crop_count,
            full_image_prepared=True,
            crop_montage_prepared=True,
            full_image_size_limit=self.image_size,
            crop_montage_size_limit=self.crop_montage_size,
            full_image_size=read_image_size_from_bytes(full_image_bytes),
            crop_montage_size=(montage_result.width, montage_result.height),
            crop_montage_path=crop_montage_path,
            image_preparation_seconds=preparation_seconds,
        )

        inference_started = perf_counter()
        try:
            try:
                response = self.client.generate(
                    prompt,
                    image_bytes_list=image_bytes_list,
                )
                self.last_ollama_metadata = getattr(self.client, "last_response_metadata", None)
                self.last_vlm_status = "success"
            except OllamaContentError as exc:
                self.last_ollama_metadata = exc.metadata
                self.last_vlm_status = _vlm_status_for_empty_content(exc.metadata)
                return self._record_fallback(response="", parse_error=str(exc), yolo_result=yolo_result)
            except ValueError as exc:
                self.last_ollama_metadata = getattr(self.client, "last_response_metadata", None)
                self.last_vlm_status = "empty_content"
                return self._record_fallback(response="", parse_error=str(exc), yolo_result=yolo_result)
            except RuntimeError as exc:
                self.last_ollama_metadata = getattr(self.client, "last_response_metadata", None)
                self.last_vlm_status = _vlm_status_for_runtime_error(exc)
                return self._record_fallback(response="", parse_error=str(exc), yolo_result=yolo_result)
        finally:
            self.last_preparation_info.inference_seconds = (
                perf_counter() - inference_started
            )
        self.last_raw_response = response
        parse_result = self.response_parser.parse_response(response, yolo_result)
        self.last_parse_result = parse_result
        self.last_parse_success = parse_result.parse_success
        self.last_parse_error = parse_result.parse_error
        self.last_fallback_used = parse_result.fallback_used
        self.last_parse_status = _parse_status_from_result(parse_result)
        self.last_quality_info = parse_result.quality_info
        return parse_result.formatted_response

    def _record_fallback(
        self,
        response: str,
        parse_error: str,
        yolo_result: YoloResult,
    ) -> str:
        parse_result = VlmParseResult(
            raw_response=response,
            parse_success=False,
            parse_error=parse_error,
            fallback_used=True,
            parsed_response=None,
            formatted_response=format_yolo_fallback_response(yolo_result),
        )
        self.last_raw_response = response
        self.last_parse_result = parse_result
        self.last_parse_success = parse_result.parse_success
        self.last_parse_error = parse_result.parse_error
        self.last_fallback_used = parse_result.fallback_used
        self.last_parse_status = "not_attempted"
        self.last_quality_info = parse_result.quality_info
        return parse_result.formatted_response

    def _save_crop_montage_if_enabled(
        self,
        montage_bytes: bytes,
        source_image_path: Path,
    ) -> Path | None:
        if not self.save_crop_montage:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_stem = _safe_filename_stem(source_image_path.stem)
        output_path = self.crop_montage_output_dir / f"{safe_stem}_crop_montage_{timestamp}.jpg"
        try:
            saved_path = save_montage_bytes(montage_bytes, output_path)
        except OSError as exc:
            print(f"[WARN] Failed to save crop montage: {exc}")
            return None

        self.last_crop_montage_path = saved_path
        print(f"[INFO] Crop montage saved: {saved_path.resolve()}")
        return saved_path

    def _image_bytes_for_mode(
        self,
        *,
        full_image_bytes: bytes,
        montage_bytes: bytes,
    ) -> list[bytes]:
        if self.image_mode == "full":
            return [full_image_bytes]
        if self.image_mode == "montage":
            return [montage_bytes]
        if self.image_mode == "full_montage":
            return [full_image_bytes, montage_bytes]
        raise ValueError(f"Unsupported VLM image mode: {self.image_mode}")


def _safe_filename_stem(stem: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)
    return safe or "image"


def _vlm_status_for_empty_content(metadata: OllamaResponseMetadata) -> str:
    if metadata.done is False:
        return "done_false"
    return "empty_content"


def _vlm_status_for_runtime_error(exc: RuntimeError) -> str:
    message = str(exc).lower()
    if "http request failed" in message or "status_code=" in message:
        return "http_error"
    if "invalid json" in message:
        return "response_json_error"
    if "ollama returned an error" in message:
        return "ollama_error"
    return "exception"


def _parse_status_from_result(parse_result: VlmParseResult) -> str:
    if parse_result.parse_success:
        return "success"
    if parse_result.parse_error.startswith("Invalid JSON"):
        return "json_parse_failed"
    return "validation_failed"

