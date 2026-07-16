from __future__ import annotations

import sys
from pathlib import Path

import pytest

from model.defect_info import Detection
from model.inspection_result import InspectionResult
import scripts.test_yolo_vlm as cli
from vlm.response_parser import VlmQualityInfo


def test_cli_vlm_defaults_and_skip_vlm_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["test_yolo_vlm.py", "--image", "sample.jpg", "--skip-vlm"]
    )

    args = cli.parse_args()

    assert args.ollama_host == "http://127.0.0.1:11434"
    assert args.vlm_num_ctx == 8192
    assert args.vlm_num_predict == 256
    assert args.vlm_temperature == 0.0
    assert args.vlm_top_p == 0.8
    assert args.vlm_top_k == 20
    assert args.vlm_repeat_penalty == 1.1
    assert args.vlm_seed == 42
    assert args.vlm_max_retries == 2
    assert args.vlm_retry_delay == 0.5
    assert args.vlm_timeout == 120.0
    assert args.vlm_debug_response is False
    assert args.vlm_image_size == 960
    assert args.vlm_full_image_size is None
    assert args.vlm_image_mode == "full_montage"
    assert args.vlm_crop_montage_size == 960
    assert args.vlm_montage_size is None
    assert args.vlm_crop_padding == 192
    assert args.vlm_crop_min_size == 256
    assert args.vlm_crop_max_size == 512
    assert args.save_crop_montage is False
    assert args.crop_montage_output_dir == "data/result_images/montage"
    assert args.iou == 0.5
    assert args.skip_vlm is True
    assert args.debug_vlm is False


def test_cli_build_vlm_service_passes_context_options(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_yolo_vlm.py",
            "--image",
            "sample.jpg",
            "--ollama-host",
            "http://example.test:11434",
            "--vlm-num-ctx",
            "10000",
            "--vlm-num-predict",
            "256",
            "--vlm-temperature",
            "0.2",
            "--vlm-top-p",
            "0.7",
            "--vlm-top-k",
            "10",
            "--vlm-repeat-penalty",
            "1.2",
            "--vlm-seed",
            "7",
            "--vlm-max-retries",
            "3",
            "--vlm-retry-delay",
            "1.5",
            "--vlm-timeout",
            "90",
            "--vlm-image-size",
            "640",
            "--vlm-full-image-size",
            "768",
            "--vlm-image-mode",
            "montage",
            "--vlm-crop-montage-size",
            "700",
            "--vlm-montage-size",
            "896",
            "--vlm-crop-padding",
            "120",
            "--vlm-crop-min-size",
            "180",
            "--vlm-crop-max-size",
            "420",
        ],
    )

    args = cli.parse_args()
    service = cli.build_vlm_service(args)

    assert service.client.host == "http://example.test:11434"
    assert service.client.num_ctx == 10000
    assert service.client.num_predict == 256
    assert service.client.temperature == 0.2
    assert service.client.top_p == 0.7
    assert service.client.top_k == 10
    assert service.client.repeat_penalty == 1.2
    assert service.client.seed == 7
    assert service.client.timeout_seconds == 90
    assert service.image_size == 768
    assert service.image_mode == "montage"
    assert service.crop_montage_size == 896
    assert service.crop_padding == 120
    assert service.crop_min_size == 180
    assert service.crop_max_size == 420
    assert service.max_retries == 3
    assert service.retry_delay_seconds == 1.5


def test_cli_detection_rows_include_location(capsys) -> None:
    cli.print_detection_rows(
        [
            Detection(
                0, "open_circuit", 0.784, 2711, 946, 2739, 979, location="middle right"
            )
        ]
    )

    output = capsys.readouterr().out

    assert "location=middle right" in output
    assert "box=(2711, 946, 2739, 979)" in output


def test_cli_help_includes_crop_montage_and_debug_options(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["test_yolo_vlm.py", "--help"])

    try:
        cli.parse_args()
    except SystemExit as exc:
        assert exc.code == 0

    output = capsys.readouterr().out
    assert "--vlm-full-image-size" in output
    assert "--vlm-montage-size" in output
    assert "--vlm-image-mode" in output
    assert "--vlm-crop-montage-size" in output
    assert "--vlm-crop-padding" in output
    assert "--vlm-crop-min-size" in output
    assert "--vlm-crop-max-size" in output
    assert "--vlm-temperature" in output
    assert "--vlm-top-p" in output
    assert "--vlm-top-k" in output
    assert "--vlm-repeat-penalty" in output
    assert "--vlm-seed" in output
    assert "--vlm-max-retries" in output
    assert "--vlm-retry-delay" in output
    assert "--vlm-timeout" in output
    assert "--vlm-debug-response" in output
    assert "--save-crop-montage" in output
    assert "--crop-montage-output-dir" in output
    assert "--debug-vlm" in output


@pytest.mark.parametrize(
    ("full_size", "montage_size"),
    [
        ("640", "640"),
        ("768", "768"),
        ("960", "960"),
    ],
)
def test_cli_accepts_vlm_size_experiment_options(monkeypatch, full_size: str, montage_size: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_yolo_vlm.py",
            "--image",
            "sample.jpg",
            "--vlm-full-image-size",
            full_size,
            "--vlm-montage-size",
            montage_size,
        ],
    )

    args = cli.parse_args()
    service = cli.build_vlm_service(args)

    assert service.image_size == int(full_size)
    assert service.crop_montage_size == int(montage_size)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--vlm-full-image-size", "0"),
        ("--vlm-full-image-size", "-1"),
        ("--vlm-full-image-size", "bad"),
        ("--vlm-montage-size", "0"),
        ("--vlm-montage-size", "-1"),
        ("--vlm-montage-size", "bad"),
        ("--vlm-max-retries", "-1"),
        ("--vlm-max-retries", "bad"),
    ],
)
def test_cli_rejects_invalid_vlm_size_experiment_options(monkeypatch, option: str, value: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["test_yolo_vlm.py", "--image", "sample.jpg", option, value],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args()

    assert exc_info.value.code == 2


def test_cli_rejects_invalid_vlm_image_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["test_yolo_vlm.py", "--image", "sample.jpg", "--vlm-image-mode", "both"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args()

    assert exc_info.value.code == 2


class FakeVlmService:
    last_preparation_info = None
    last_raw_response = '{"raw": true}'
    last_quality_info = VlmQualityInfo()


class FakeInspectionService:
    def __init__(self, yolo_service: object, vlm_service: object) -> None:
        self.yolo_service = yolo_service
        self.vlm_service = vlm_service

    def inspect(self, image_path: Path) -> InspectionResult:
        return InspectionResult(
            source_image_path=image_path,
            result_image_path=Path("result.jpg"),
            status="NG",
            detections=[Detection(0, "open_circuit", 0.8, 1, 2, 3, 4)],
            vlm_explanation="sanitized explanation",
        )


def install_main_fakes(monkeypatch) -> None:
    monkeypatch.setattr(cli, "build_yolo_service", lambda args: object())
    monkeypatch.setattr(cli, "build_vlm_service", lambda args: FakeVlmService())
    monkeypatch.setattr(cli, "InspectionService", FakeInspectionService)


def test_cli_default_output_hides_raw_vlm_response(monkeypatch, capsys) -> None:
    install_main_fakes(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["test_yolo_vlm.py", "--image", "sample.jpg"])

    assert cli.main() == 0

    output = capsys.readouterr().out
    assert "sanitized explanation" in output
    assert "[VLM raw response]" not in output
    assert '{"raw": true}' not in output


def test_cli_debug_vlm_prints_raw_vlm_response(monkeypatch, capsys) -> None:
    install_main_fakes(monkeypatch)
    monkeypatch.setattr(
        sys, "argv", ["test_yolo_vlm.py", "--image", "sample.jpg", "--debug-vlm"]
    )

    assert cli.main() == 0

    output = capsys.readouterr().out
    assert "sanitized explanation" in output
    assert "[VLM raw response]" in output
    assert '{"raw": true}' in output
