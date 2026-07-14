from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from time import perf_counter, sleep

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


ZERO_VALUE_RETRY_IMAGE_SIZE = 640


@dataclass(slots=True)
class VlmImageDiagnostic:
    label: str
    byte_length: int
    sha256: str
    decoded_size: tuple[int, int]
    base64_length: int


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
    prompt_character_count: int = 0
    request_json_size: int | None = None
    image_diagnostics: tuple[VlmImageDiagnostic, ...] = ()
    zero_value_recovery_used: bool = False
    zero_value_recovery_image_size: int | None = None
    zero_value_unload_succeeded: bool | None = None


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
        max_retries: int = 0,
        retry_delay_seconds: float = 0.0,
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
        self.max_retries = max(0, int(max_retries))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
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
        self.last_retry_count: int = 0
        self.last_failure_reason: str = ""
        self.last_failure_reasons: list[str] = []
        self.last_error_type: str = ""
        self.last_error_message: str = ""

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
        self.last_retry_count = 0
        self.last_failure_reason = ""
        self.last_failure_reasons = []
        self.last_error_type = ""
        self.last_error_message = ""
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
        image_labels = self._image_labels_for_mode()

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
            prompt_character_count=len(prompt),
            image_diagnostics=_build_image_diagnostics(image_labels, image_bytes_list),
        )

        inference_started = perf_counter()
        max_attempts = self.max_retries + 1
        current_image_bytes_list = image_bytes_list
        zero_value_recovery_used = False
        try:
            for attempt_index in range(max_attempts):
                attempt_number = attempt_index + 1
                if attempt_index:
                    self.last_retry_count = attempt_index
                    self.last_vlm_status = "retrying"
                    if self.retry_delay_seconds:
                        print(f"[INFO] Retrying after {self.retry_delay_seconds:g} seconds")
                        sleep(self.retry_delay_seconds)
                print(f"[INFO] VLM attempt {attempt_number}/{max_attempts} started")
                print(f"[INFO] Ollama endpoint: {getattr(self.client, 'endpoint', '')}")
                print(f"[INFO] Ollama stream: {str(getattr(self.client, 'stream', None)).lower()}")

                try:
                    response = self.client.generate(
                        prompt,
                        image_bytes_list=current_image_bytes_list,
                    )
                    self.last_ollama_metadata = getattr(self.client, "last_response_metadata", None)
                    if self.last_preparation_info is not None:
                        request_summary = getattr(self.client, "last_request_summary", None)
                        if request_summary is not None:
                            self.last_preparation_info.request_json_size = request_summary.get("json_size_bytes")
                except OllamaContentError as exc:
                    self.last_ollama_metadata = exc.metadata
                    self.last_vlm_status = _vlm_status_for_empty_content(exc.metadata)
                    self.last_failure_reason = self.last_vlm_status
                    self.last_failure_reasons.append(self.last_failure_reason)
                    self.last_error_type = self.last_vlm_status
                    self.last_error_message = str(exc)
                    self._print_ollama_metadata()
                    print(f"[INFO] VLM attempt {attempt_number}/{max_attempts} failed: {self.last_vlm_status}")
                    if (
                        _is_zero_value_response(exc.metadata)
                        and attempt_index < self.max_retries
                    ):
                        if not zero_value_recovery_used:
                            unload_succeeded = _unload_client_model(self.client)
                            recovered_images = self._build_zero_value_retry_images(
                                image_path=image_path,
                                yolo_result=yolo_result,
                            )
                            current_image_bytes_list = recovered_images
                            zero_value_recovery_used = True
                            if self.last_preparation_info is not None:
                                self.last_preparation_info.zero_value_recovery_used = True
                                self.last_preparation_info.zero_value_recovery_image_size = (
                                    ZERO_VALUE_RETRY_IMAGE_SIZE
                                )
                                self.last_preparation_info.zero_value_unload_succeeded = unload_succeeded
                                self.last_preparation_info.image_count = len(recovered_images)
                                self.last_preparation_info.image_diagnostics = _build_image_diagnostics(
                                    image_labels,
                                    recovered_images,
                                )
                                sizes = [diagnostic.decoded_size for diagnostic in self.last_preparation_info.image_diagnostics]
                                if self.image_mode in {"full", "full_montage"} and sizes:
                                    self.last_preparation_info.full_image_size = sizes[0]
                                if self.image_mode == "montage" and sizes:
                                    self.last_preparation_info.crop_montage_size = sizes[0]
                                elif self.image_mode == "full_montage" and len(sizes) > 1:
                                    self.last_preparation_info.crop_montage_size = sizes[1]
                            print(
                                "[INFO] Zero-value Ollama response detected; "
                                f"unload_succeeded={str(unload_succeeded).lower()}, "
                                f"retry_image_size={ZERO_VALUE_RETRY_IMAGE_SIZE}"
                            )
                        else:
                            unload_succeeded = _unload_client_model(self.client)
                            if self.last_preparation_info is not None:
                                self.last_preparation_info.zero_value_unload_succeeded = unload_succeeded
                            print(
                                "[INFO] Repeated zero-value Ollama response; "
                                f"unload_succeeded={str(unload_succeeded).lower()}"
                            )
                        continue
                    if attempt_index < self.max_retries:
                        continue
                    return self._record_fallback(response="", parse_error=str(exc), yolo_result=yolo_result)
                except ValueError as exc:
                    self.last_ollama_metadata = getattr(self.client, "last_response_metadata", None)
                    self.last_vlm_status = "empty_content"
                    self.last_failure_reason = self.last_vlm_status
                    self.last_failure_reasons.append(self.last_failure_reason)
                    self.last_error_type = self.last_vlm_status
                    self.last_error_message = str(exc)
                    self._print_ollama_metadata()
                    print(f"[INFO] VLM attempt {attempt_number}/{max_attempts} failed: {self.last_vlm_status}")
                    if attempt_index < self.max_retries:
                        continue
                    return self._record_fallback(response="", parse_error=str(exc), yolo_result=yolo_result)
                except RuntimeError as exc:
                    self.last_ollama_metadata = getattr(self.client, "last_response_metadata", None)
                    self.last_vlm_status = _vlm_status_for_runtime_error(exc)
                    self.last_failure_reason = self.last_vlm_status
                    self.last_failure_reasons.append(self.last_failure_reason)
                    self.last_error_type = self.last_vlm_status
                    self.last_error_message = str(exc)
                    self._print_ollama_metadata()
                    print(f"[INFO] VLM attempt {attempt_number}/{max_attempts} failed: {self.last_vlm_status}")
                    if attempt_index < self.max_retries:
                        continue
                    return self._record_fallback(response="", parse_error=str(exc), yolo_result=yolo_result)

                self.last_raw_response = response
                self._print_ollama_metadata()
                parse_result = self.response_parser.parse_response(response, yolo_result)
                self.last_parse_result = parse_result
                self.last_parse_success = parse_result.parse_success
                self.last_parse_error = parse_result.parse_error
                self.last_fallback_used = parse_result.fallback_used
                self.last_parse_status = _parse_status_from_result(parse_result)
                self.last_quality_info = parse_result.quality_info
                if parse_result.parse_success:
                    self.last_vlm_status = "retry_success" if attempt_index else "success"
                    self.last_failure_reason = "|".join(self.last_failure_reasons)
                    self.last_error_type = ""
                    self.last_error_message = ""
                    print(f"[INFO] VLM attempt {attempt_number}/{max_attempts} succeeded")
                    return parse_result.formatted_response

                self.last_vlm_status = self.last_parse_status
                self.last_failure_reason = self.last_parse_status
                self.last_failure_reasons.append(self.last_failure_reason)
                self.last_error_type = self.last_parse_status
                self.last_error_message = parse_result.parse_error
                print(f"[INFO] VLM attempt {attempt_number}/{max_attempts} failed: {self.last_parse_status}")
                if attempt_index < self.max_retries:
                    continue
                self.last_vlm_status = "success"
                return parse_result.formatted_response

            return self._record_fallback(
                response=self.last_raw_response or "",
                parse_error=self.last_failure_reason or "VLM retry attempts exhausted.",
                yolo_result=yolo_result,
            )
        finally:
            self.last_preparation_info.inference_seconds = (
                perf_counter() - inference_started
            )

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

    def _print_ollama_metadata(self) -> None:
        metadata = self.last_ollama_metadata
        if metadata is None:
            print("[INFO] Ollama HTTP status: ")
            print("[INFO] Ollama done: ")
            print("[INFO] Ollama done_reason: ")
            print("[INFO] Ollama response length: ")
            return
        print(f"[INFO] Ollama HTTP status: {_format_optional(metadata.http_status)}")
        print(f"[INFO] Ollama done: {_format_optional(metadata.done)}")
        print(f"[INFO] Ollama done_reason: {_format_optional(metadata.done_reason)}")
        print(f"[INFO] Ollama error: {_format_optional(metadata.error)}")
        print(f"[INFO] Ollama response length: {_format_optional(metadata.content_length)}")
        print(f"[INFO] Ollama prompt eval count: {_format_optional(metadata.prompt_eval_count)}")
        print(f"[INFO] Ollama eval count: {_format_optional(metadata.eval_count)}")
        print(f"[INFO] Ollama total duration: {_format_optional(metadata.total_duration)}")
        print(f"[INFO] Ollama load duration: {_format_optional(metadata.load_duration)}")
        print(f"[INFO] Ollama prompt eval duration: {_format_optional(metadata.prompt_eval_duration)}")
        print(f"[INFO] Ollama eval duration: {_format_optional(metadata.eval_duration)}")

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

    def _image_labels_for_mode(self) -> list[str]:
        if self.image_mode == "full":
            return ["full"]
        if self.image_mode == "montage":
            return ["montage"]
        if self.image_mode == "full_montage":
            return ["full", "montage"]
        raise ValueError(f"Unsupported VLM image mode: {self.image_mode}")

    def _build_zero_value_retry_images(
        self,
        *,
        image_path: Path,
        yolo_result: YoloResult,
    ) -> list[bytes]:
        retry_size = min(self.image_size, ZERO_VALUE_RETRY_IMAGE_SIZE)
        retry_montage_size = min(self.crop_montage_size, ZERO_VALUE_RETRY_IMAGE_SIZE)
        full_image_bytes = resize_image_to_jpeg_bytes(
            image_path,
            max_size=retry_size,
            quality=self.image_quality,
        )
        montage_result = create_crop_montage_result(
            image_path=image_path,
            detections=yolo_result.detections,
            max_size=retry_montage_size,
            quality=self.image_quality,
            padding=self.crop_padding,
            min_crop_size=self.crop_min_size,
            max_crop_size=self.crop_max_size,
        )
        return self._image_bytes_for_mode(
            full_image_bytes=full_image_bytes,
            montage_bytes=montage_result.image_bytes,
        )


def _safe_filename_stem(stem: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)
    return safe or "image"


def _format_optional(value: object | None) -> str:
    return "" if value is None else str(value)


def _build_image_diagnostics(
    labels: list[str],
    image_bytes_list: list[bytes],
) -> tuple[VlmImageDiagnostic, ...]:
    diagnostics: list[VlmImageDiagnostic] = []
    for label, image_bytes in zip(labels, image_bytes_list, strict=True):
        diagnostics.append(
            VlmImageDiagnostic(
                label=label,
                byte_length=len(image_bytes),
                sha256=hashlib.sha256(image_bytes).hexdigest(),
                decoded_size=read_image_size_from_bytes(image_bytes),
                base64_length=len(base64.b64encode(image_bytes)),
            )
        )
    return tuple(diagnostics)


def _is_zero_value_response(metadata: OllamaResponseMetadata) -> bool:
    return metadata.done is False and metadata.content_length == 0


def _unload_client_model(client: object) -> bool:
    unload = getattr(client, "unload_model", None)
    if not callable(unload):
        return False
    try:
        return bool(unload())
    except Exception:
        return False


def _vlm_status_for_empty_content(metadata: OllamaResponseMetadata) -> str:
    if metadata.done is False:
        return "done_false"
    return "empty_content"


def _vlm_status_for_runtime_error(exc: RuntimeError) -> str:
    message = str(exc).lower()
    if "timed out" in message or "timeout" in message:
        return "timeout"
    if "http request failed" in message or "status_code=" in message:
        return "http_error"
    if "failed to connect" in message:
        return "connection_error"
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

