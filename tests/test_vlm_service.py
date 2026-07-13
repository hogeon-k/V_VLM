from pathlib import Path

from PIL import Image

from model.defect_info import Detection
from model.yolo_result import YoloResult
from service.vlm_service import VlmService


def test_vlm_service_skips_ok_result() -> None:
    description = VlmService().describe_defects(Path("sample.png"), YoloResult(Path("sample.png")))

    assert description is None


def test_vlm_service_generates_description_for_ng_result(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    class FakeClient:
        def generate(self, prompt: str, image_bytes_list: list[bytes]) -> str:
            assert "open_circuit" in prompt
            assert "위치 미계산" in prompt
            assert len(image_bytes_list) == 2
            assert all(image_bytes_list)
            return "  explanation  "

    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4)
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])

    service = VlmService(client=FakeClient())
    description = service.describe_defects(image_path, yolo_result)

    assert description == "explanation"


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
    ) -> bytes:
        calls["montage"] = (
            image_path,
            list(detections),
            max_size,
            quality,
            padding,
            min_crop_size,
            max_crop_size,
        )
        return b"montage-image"

    def fake_size(image_bytes: bytes) -> tuple[int, int]:
        return (960, 240) if image_bytes == b"resized-image" else (512, 512)

    class FakeClient:
        def generate(
            self,
            prompt: str,
            image_bytes_list: list[bytes] | None = None,
        ) -> str:
            calls["client"] = image_bytes_list
            return "explanation"

    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", fake_resize)
    monkeypatch.setattr("service.vlm_service.create_crop_montage_jpeg_bytes", fake_montage)
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

    assert description == "explanation"
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
    assert service.last_preparation_info.full_image_size == (960, 240)
    assert service.last_preparation_info.crop_montage_size == (512, 512)


def test_vlm_service_prompt_includes_calculated_location(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_resize(image_path: Path, max_size: int, quality: int) -> bytes:
        return b"resized-image"

    class FakeClient:
        def generate(self, prompt: str, image_bytes_list: list[bytes] | None = None) -> str:
            calls["prompt"] = prompt
            return "explanation"

    monkeypatch.setattr("service.vlm_service.resize_image_to_jpeg_bytes", fake_resize)
    monkeypatch.setattr(
        "service.vlm_service.create_crop_montage_jpeg_bytes",
        lambda **kwargs: b"montage-image",
    )
    monkeypatch.setattr(
        "service.vlm_service.read_image_size_from_bytes",
        lambda image_bytes: (100, 80),
    )

    detection = Detection(0, "open_circuit", 0.91, 1, 2, 3, 4, location="중단 오른쪽")
    yolo_result = YoloResult(Path("sample.png"), detections=[detection])

    VlmService(client=FakeClient()).describe_defects(Path("result.jpg"), yolo_result)

    assert "- 위치: 중단 오른쪽" in str(calls["prompt"])
