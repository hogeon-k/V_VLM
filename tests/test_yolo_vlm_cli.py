from __future__ import annotations

import sys
from pathlib import Path

from model.defect_info import Detection
from model.inspection_result import InspectionResult
import scripts.test_yolo_vlm as cli


def test_cli_vlm_defaults_and_skip_vlm_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["test_yolo_vlm.py", "--image", "sample.jpg", "--skip-vlm"]
    )

    args = cli.parse_args()

    assert args.ollama_host == "http://127.0.0.1:11434"
    assert args.vlm_num_ctx == 8192
    assert args.vlm_num_predict == 512
    assert args.vlm_image_size == 960
    assert args.vlm_crop_montage_size == 960
    assert args.vlm_crop_padding == 192
    assert args.vlm_crop_min_size == 256
    assert args.vlm_crop_max_size == 512
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
            "--vlm-image-size",
            "640",
            "--vlm-crop-montage-size",
            "700",
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
    assert service.image_size == 640
    assert service.crop_montage_size == 700
    assert service.crop_padding == 120
    assert service.crop_min_size == 180
    assert service.crop_max_size == 420


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
    assert "--vlm-crop-montage-size" in output
    assert "--vlm-crop-padding" in output
    assert "--vlm-crop-min-size" in output
    assert "--vlm-crop-max-size" in output
    assert "--debug-vlm" in output


class FakeVlmService:
    last_preparation_info = None
    last_raw_response = '{"raw": true}'


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