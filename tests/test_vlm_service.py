import json
from pathlib import Path

import pytest
from PIL import Image

from model.defect_info import Detection
from model.yolo_result import YoloResult
from service.vlm_service import VlmService
from vlm.crop_montage import CropMontageResult
from vlm.ollama_response import OllamaContentError, OllamaResponseMetadata


def valid_response() -> str:
    return json.dumps(
        {
            "final_judgment": "NG",
            "detections": [
                {
                    "detection_id": 1,
                    "visual_feature": "패턴 경계가 중간에서 불연속적으로 보입니다.",
                    "visibility": "clear",
                    "review_required": False,
                }
            ],
            "summary": "총 1개의 결함이 탐지되었으며, 1개는 시각적 특징이 명확합니다.",
        }
    )


def test_vlm_service_skips_ok_result() -> None:
    description = VlmService().describe_defects(Path("sample.png"), YoloResult(Path("sample.png")))

    assert description is None


def test_vlm_service_generates_structured_description_for_ng_result(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    class FakeClient:
        def __init__(self) -> None:
            self.unload_calls = 0

        def unload_model(self) -> bool:
            self.unload_calls += 1
            return True

        def generate(self, prompt: str, image_bytes_list: list[bytes]) -> str:
            assert "open_circuit" in prompt
            assert "위치: 위치 정보 없음" in prompt
            assert len(image_bytes_list) == 2
            assert all(image_bytes_list)
            return valid_response()

    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])

    client = FakeClient()
    service = VlmService(client=client)
    description = service.describe_defects(image_path, yolo_result)

    assert description is not None
    assert "최종 판정: NG" in description
    assert "탐지된 불량 수: 1개" in description
    assert "패턴 경계가 중간에서 불연속적으로 보입니다." in description
    assert service.last_raw_response == valid_response()
    assert service.last_parse_success is True
    assert service.last_fallback_used is False
    assert service.last_parse_error == ""
    assert service.last_vlm_status == "success"
    assert service.last_parse_status == "success"
    assert service.last_quality_info.quality_status == "acceptable"
    assert client.unload_calls == 1
    assert service.last_preparation_info is not None
    assert service.last_preparation_info.final_unload_succeeded is True


def test_vlm_service_passes_full_image_and_montage_bytes(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_resize(image_path: Path, max_size: int, quality: int) -> bytes:
        calls["resize"] = (image_path, max_size, quality)
        return b"resized-image"

    def fake_montage(
        image_path: Path,
        detections: object,
        max_size: int,
        quality: int,
        padding: int,
        min_crop_size: int,
        max_crop_size: int,
    ) -> CropMontageResult:
        calls["montage"] = (
            image_path,
            list(detections),
            max_size,
            quality,
            padding,
            min_crop_size,
            max_crop_size,
        )
        return CropMontageResult(b"montage-image", 512, 512, 1)

    def fake_size(image_bytes: bytes) -> tuple[int, int]:
        return (960, 240) if image_bytes == b"resized-image" else (512, 512)

    class FakeClient:
        def generate(
            self,
            prompt: str,
            image_bytes_list: list[bytes] | None = None,
        ) -> str:
            calls["client"] = image_bytes_list
            return valid_response()

    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", fake_resize)
    monkeypatch.setattr("service.vlm_service.create_crop_montage_result", fake_montage)
    monkeypatch.setattr("service.vlm_service.read_image_size_from_bytes", fake_size)

    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])
    service = VlmService(
        client=FakeClient(),
        image_size=512,
        image_quality=85,
        crop_montage_size=640,
        crop_padding=100,
        crop_min_size=128,
        crop_max_size=300,
    )

    description = service.describe_defects(Path("result.jpg"), yolo_result)

    assert description is not None
    assert "최종 판정: NG" in description
    assert calls["resize"] == (Path("result.jpg"), 512, 85)
    assert calls["montage"] == (
        Path("result.jpg"),
        [detection],
        640,
        85,
        100,
        128,
        300,
    )
    assert calls["client"] == [b"resized-image", b"montage-image"]
    assert service.last_preparation_info is not None
    assert service.last_preparation_info.image_count == 2
    assert service.last_preparation_info.detection_crop_count == 1
    assert service.last_preparation_info.full_image_size_limit == 512
    assert service.last_preparation_info.crop_montage_size_limit == 640
    assert service.last_preparation_info.full_image_size == (960, 240)
    assert service.last_preparation_info.crop_montage_size == (512, 512)


@pytest.mark.parametrize(
    ("image_mode", "expected_images"),
    [
        ("full", [b"full-image"]),
        ("montage", [b"montage-image"]),
        ("full_montage", [b"full-image", b"montage-image"]),
    ],
)
def test_vlm_service_selects_images_by_mode(
    monkeypatch,
    image_mode: str,
    expected_images: list[bytes],
) -> None:
    calls: dict[str, object] = {}

    class FakeClient:
        def generate(self, prompt: str, image_bytes_list: list[bytes] | None = None) -> str:
            calls["images"] = image_bytes_list
            return valid_response()

    monkeypatch.setattr(
        "service.vlm_service.resize_image_to_jpeg_bytes",
        lambda *args, **kwargs: b"full-image",
    )
    monkeypatch.setattr(
        "service.vlm_service.create_crop_montage_result",
        lambda **kwargs: CropMontageResult(b"montage-image", 320, 240, 1),
    )
    monkeypatch.setattr("service.vlm_service.read_image_size_from_bytes", lambda image_bytes: (640, 335))

    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])
    service = VlmService(client=FakeClient(), image_mode=image_mode)

    service.describe_defects(Path("result.jpg"), yolo_result)

    assert calls["images"] == expected_images
    assert service.last_preparation_info is not None
    assert service.last_preparation_info.image_mode == image_mode
    assert service.last_preparation_info.image_count == len(expected_images)
    assert service.last_preparation_info.full_image_prepared is True
    assert service.last_preparation_info.crop_montage_prepared is True
    assert service.last_preparation_info.full_image_size == (640, 335)
    assert service.last_preparation_info.crop_montage_size == (320, 240)


def test_vlm_service_prompt_includes_calculated_location(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeClient:
        def generate(self, prompt: str, image_bytes_list: list[bytes] | None = None) -> str:
            calls["prompt"] = prompt
            return valid_response()

    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", lambda *args, **kwargs: b"image")
    monkeypatch.setattr(
        "service.vlm_service.create_crop_montage_result",
        lambda **kwargs: CropMontageResult(b"montage-image", 100, 80, 1),
    )
    monkeypatch.setattr("service.vlm_service.read_image_size_from_bytes", lambda image_bytes: (100, 80))

    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4, location="middle right")
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])

    VlmService(client=FakeClient()).describe_defects(Path("result.jpg"), yolo_result)

    assert "위치: middle right" in str(calls["prompt"])


def test_vlm_service_saves_montage_when_enabled(monkeypatch, tmp_path, capsys) -> None:
    class FakeClient:
        def generate(self, prompt: str, image_bytes_list: list[bytes] | None = None) -> str:
            return "not json"

    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", lambda *args, **kwargs: b"image")
    monkeypatch.setattr(
        "service.vlm_service.create_crop_montage_result",
        lambda **kwargs: CropMontageResult(b"montage-image", 320, 240, 1),
    )
    monkeypatch.setattr("service.vlm_service.read_image_size_from_bytes", lambda image_bytes: (100, 80))

    output_dir = tmp_path / "montage"
    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample board.png"), detections=[detection])
    service = VlmService(
        client=FakeClient(),
        save_crop_montage=True,
        crop_montage_output_dir=output_dir,
    )

    service.describe_defects(Path("result.jpg"), yolo_result)

    saved_files = list(output_dir.glob("sample_board_crop_montage_*.jpg"))
    assert len(saved_files) == 1
    assert saved_files[0].read_bytes() == b"montage-image"
    assert service.last_preparation_info is not None
    assert service.last_preparation_info.crop_montage_path == saved_files[0]
    assert service.last_parse_success is False
    assert service.last_fallback_used is True
    assert "Crop montage saved:" in capsys.readouterr().out


def test_vlm_service_retains_raw_response_and_parse_error_on_fallback(monkeypatch) -> None:
    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", lambda *args, **kwargs: b"image")
    monkeypatch.setattr(
        "service.vlm_service.create_crop_montage_result",
        lambda **kwargs: CropMontageResult(b"montage-image", 320, 240, 1),
    )
    monkeypatch.setattr("service.vlm_service.read_image_size_from_bytes", lambda image_bytes: (100, 80))

    class FakeClient:
        def generate(self, prompt: str, image_bytes_list: list[bytes] | None = None) -> str:
            return "not json"

    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])
    service = VlmService(client=FakeClient())

    description = service.describe_defects(Path("result.jpg"), yolo_result)

    assert description is not None
    assert "최종 판정: NG" in description
    assert service.last_raw_response == "not json"
    assert service.last_parse_success is False
    assert service.last_fallback_used is True
    assert service.last_parse_error.startswith("Invalid JSON")
    assert service.last_vlm_status == "success"
    assert service.last_parse_status == "json_parse_failed"


def test_vlm_service_converts_empty_assistant_content_error_to_fallback(monkeypatch) -> None:
    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", lambda *args, **kwargs: b"image")
    monkeypatch.setattr(
        "service.vlm_service.create_crop_montage_result",
        lambda **kwargs: CropMontageResult(b"montage-image", 320, 240, 1),
    )
    monkeypatch.setattr("service.vlm_service.read_image_size_from_bytes", lambda image_bytes: (100, 80))

    class FakeClient:
        last_response_metadata = OllamaResponseMetadata(done=False, content_length=0)

        def generate(self, prompt: str, image_bytes_list: list[bytes] | None = None) -> str:
            raise OllamaContentError(
                "Ollama response JSON did not contain assistant content.",
                self.last_response_metadata,
            )

    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])
    service = VlmService(client=FakeClient())

    description = service.describe_defects(Path("result.jpg"), yolo_result)

    assert description is not None
    assert service.last_raw_response == ""
    assert service.last_parse_success is False
    assert service.last_fallback_used is True
    assert service.last_parse_error == "Ollama response JSON did not contain assistant content."
    assert service.last_vlm_status == "done_false"
    assert service.last_parse_status == "not_attempted"
    assert service.last_ollama_metadata is not None
    assert service.last_ollama_metadata.done is False
    assert service.last_ollama_metadata.content_length == 0


def test_vlm_service_records_schema_validation_failure_status(monkeypatch) -> None:
    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", lambda *args, **kwargs: b"image")
    monkeypatch.setattr(
        "service.vlm_service.create_crop_montage_result",
        lambda **kwargs: CropMontageResult(b"montage-image", 320, 240, 1),
    )
    monkeypatch.setattr("service.vlm_service.read_image_size_from_bytes", lambda image_bytes: (100, 80))

    class FakeClient:
        def generate(self, prompt: str, image_bytes_list: list[bytes] | None = None) -> str:
            return '{"final_judgment": "MAYBE", "detections": [], "summary": ""}'

    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])
    service = VlmService(client=FakeClient())

    service.describe_defects(Path("result.jpg"), yolo_result)

    assert service.last_vlm_status == "success"
    assert service.last_parse_status == "validation_failed"
    assert service.last_fallback_used is True
    assert service.last_quality_info.quality_status == "not_evaluated"


def test_vlm_service_retries_parse_failure_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", lambda *args, **kwargs: b"image")
    monkeypatch.setattr(
        "service.vlm_service.create_crop_montage_result",
        lambda **kwargs: CropMontageResult(b"montage-image", 320, 240, 1),
    )
    monkeypatch.setattr("service.vlm_service.read_image_size_from_bytes", lambda image_bytes: (100, 80))
    responses = iter(["not json", valid_response()])

    class FakeClient:
        def generate(self, prompt: str, image_bytes_list: list[bytes] | None = None) -> str:
            return next(responses)

    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])
    service = VlmService(client=FakeClient(), max_retries=1, retry_delay_seconds=0)

    service.describe_defects(Path("result.jpg"), yolo_result)

    assert service.last_parse_success is True
    assert service.last_fallback_used is False
    assert service.last_vlm_status == "retry_success"
    assert service.last_retry_count == 1
    assert service.last_failure_reason == "json_parse_failed"


def test_vlm_service_retries_done_false_and_timeout_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", lambda *args, **kwargs: b"image")
    monkeypatch.setattr(
        "service.vlm_service.create_crop_montage_result",
        lambda **kwargs: CropMontageResult(b"montage-image", 320, 240, 1),
    )
    monkeypatch.setattr("service.vlm_service.read_image_size_from_bytes", lambda image_bytes: (100, 80))

    class FakeClient:
        endpoint = "/api/chat"
        stream = False

        def __init__(self) -> None:
            self.calls = 0
            self.last_response_metadata = None

        def generate(self, prompt: str, image_bytes_list: list[bytes] | None = None) -> str:
            self.calls += 1
            if self.calls == 1:
                self.last_response_metadata = OllamaResponseMetadata(
                    http_status=200,
                    endpoint="/api/chat",
                    stream=False,
                    done=False,
                    content_length=0,
                )
                raise OllamaContentError("empty", self.last_response_metadata)
            if self.calls == 2:
                raise RuntimeError("Ollama HTTP request timed out after 120s")
            self.last_response_metadata = OllamaResponseMetadata(
                http_status=200,
                endpoint="/api/chat",
                stream=False,
                done=True,
                content_length=len(valid_response()),
            )
            return valid_response()

    client = FakeClient()
    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])
    service = VlmService(client=client, max_retries=2, retry_delay_seconds=0)

    service.describe_defects(Path("result.jpg"), yolo_result)

    assert client.calls == 3
    assert service.last_parse_success is True
    assert service.last_fallback_used is False
    assert service.last_vlm_status == "retry_success"
    assert service.last_retry_count == 2
    assert service.last_failure_reason == "done_false|timeout"


def test_vlm_service_downsizes_and_unloads_after_zero_value_response(monkeypatch) -> None:
    calls: dict[str, object] = {"images": []}

    def fake_resize(image_path: Path, max_size: int, quality: int) -> bytes:
        return f"full-{max_size}".encode()

    def fake_montage(
        image_path: Path,
        detections: object,
        max_size: int,
        quality: int,
        padding: int,
        min_crop_size: int,
        max_crop_size: int,
    ) -> CropMontageResult:
        return CropMontageResult(f"montage-{max_size}".encode(), max_size, max_size, 1)

    def fake_size(image_bytes: bytes) -> tuple[int, int]:
        if image_bytes in {b"full-960", b"montage-960"}:
            return (960, 960)
        return (640, 640)

    class FakeClient:
        endpoint = "/api/chat"
        stream = False

        def __init__(self) -> None:
            self.calls = 0
            self.unload_calls = 0
            self.last_response_metadata = None

        def unload_model(self) -> bool:
            self.unload_calls += 1
            return True

        def generate(self, prompt: str, image_bytes_list: list[bytes] | None = None) -> str:
            self.calls += 1
            calls["images"].append(image_bytes_list)
            if self.calls == 1:
                self.last_response_metadata = OllamaResponseMetadata(
                    http_status=200,
                    endpoint="/api/chat",
                    stream=False,
                    done=False,
                    content_length=0,
                )
                raise OllamaContentError("empty", self.last_response_metadata)
            self.last_response_metadata = OllamaResponseMetadata(
                http_status=200,
                endpoint="/api/chat",
                stream=False,
                done=True,
                content_length=len(valid_response()),
            )
            return valid_response()

    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", fake_resize)
    monkeypatch.setattr("service.vlm_service.create_crop_montage_result", fake_montage)
    monkeypatch.setattr("service.vlm_service.read_image_size_from_bytes", fake_size)

    client = FakeClient()
    detection = Detection(0, "missing_hole", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])
    service = VlmService(
        client=client,
        image_size=960,
        crop_montage_size=960,
        image_mode="full_montage",
        max_retries=1,
        retry_delay_seconds=0,
    )

    service.describe_defects(Path("result.jpg"), yolo_result)

    assert client.calls == 2
    assert client.unload_calls == 2
    assert calls["images"][0] == [b"full-960", b"montage-960"]
    assert calls["images"][1] == [b"full-640", b"montage-640"]
    assert service.last_parse_success is True
    assert service.last_fallback_used is False
    assert service.last_preparation_info is not None
    assert service.last_preparation_info.zero_value_recovery_used is True
    assert service.last_preparation_info.zero_value_recovery_image_size == 640
    assert service.last_preparation_info.zero_value_unload_succeeded is True
    assert service.last_preparation_info.final_unload_succeeded is True
    assert service.last_preparation_info.full_image_size == (640, 640)
    assert service.last_preparation_info.crop_montage_size == (640, 640)


def test_vlm_service_uses_fallback_after_retry_exhaustion(monkeypatch) -> None:
    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", lambda *args, **kwargs: b"image")
    monkeypatch.setattr(
        "service.vlm_service.create_crop_montage_result",
        lambda **kwargs: CropMontageResult(b"montage-image", 320, 240, 1),
    )
    monkeypatch.setattr("service.vlm_service.read_image_size_from_bytes", lambda image_bytes: (100, 80))

    class FakeClient:
        endpoint = "/api/chat"
        stream = False

        def __init__(self) -> None:
            self.calls = 0
            self.last_response_metadata = OllamaResponseMetadata(
                http_status=200,
                endpoint="/api/chat",
                stream=False,
                done=False,
                content_length=0,
            )

        def generate(self, prompt: str, image_bytes_list: list[bytes] | None = None) -> str:
            self.calls += 1
            raise OllamaContentError("empty", self.last_response_metadata)

    client = FakeClient()
    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])
    service = VlmService(client=client, max_retries=2, retry_delay_seconds=0)

    service.describe_defects(Path("result.jpg"), yolo_result)

    assert client.calls == 3
    assert service.last_parse_success is False
    assert service.last_fallback_used is True
    assert service.last_vlm_status == "done_false"
    assert service.last_retry_count == 2


def test_vlm_service_records_quality_warning_for_class_name_only(monkeypatch) -> None:
    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", lambda *args, **kwargs: b"image")
    monkeypatch.setattr(
        "service.vlm_service.create_crop_montage_result",
        lambda **kwargs: CropMontageResult(b"montage-image", 320, 240, 1),
    )
    monkeypatch.setattr("service.vlm_service.read_image_size_from_bytes", lambda image_bytes: (100, 80))

    class FakeClient:
        def generate(self, prompt: str, image_bytes_list: list[bytes] | None = None) -> str:
            return json.dumps(
                {
                    "final_judgment": "NG",
                    "detections": [
                        {
                            "detection_id": 1,
                            "visual_feature": "open_circuit",
                            "visibility": "clear",
                            "review_required": False,
                        }
                    ],
                    "summary": "One defect is visible.",
                }
            )

    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])
    service = VlmService(client=FakeClient())

    service.describe_defects(Path("result.jpg"), yolo_result)

    assert service.last_parse_success is True
    assert service.last_parse_status == "success"
    assert service.last_quality_info.quality_status == "warning"
    assert service.last_quality_info.class_name_only_count == 1


def test_vlm_service_records_client_exception_as_fallback(monkeypatch) -> None:
    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", lambda *args, **kwargs: b"image")
    monkeypatch.setattr(
        "service.vlm_service.create_crop_montage_result",
        lambda **kwargs: CropMontageResult(b"montage-image", 320, 240, 1),
    )
    monkeypatch.setattr("service.vlm_service.read_image_size_from_bytes", lambda image_bytes: (100, 80))

    class FakeClient:
        def generate(self, prompt: str, image_bytes_list: list[bytes] | None = None) -> str:
            raise RuntimeError("Ollama HTTP request failed. status_code=500. body=error")

    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])
    service = VlmService(client=FakeClient())

    description = service.describe_defects(Path("result.jpg"), yolo_result)

    assert description is not None
    assert service.last_vlm_status == "http_error"
    assert service.last_parse_status == "not_attempted"
    assert service.last_fallback_used is True
