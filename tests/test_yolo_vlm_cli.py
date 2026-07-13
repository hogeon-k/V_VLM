from __future__ import annotations

import sys

from model.defect_info import Detection
import scripts.test_yolo_vlm as cli


def test_cli_vlm_defaults_and_skip_vlm_flag(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["test_yolo_vlm.py", "--image", "sample.jpg", "--skip-vlm"])

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
        [Detection(0, "open_circuit", 0.784, 2711, 946, 2739, 979, location="중단 오른쪽")]
    )

    output = capsys.readouterr().out

    assert "location=중단 오른쪽" in output
    assert "box=(2711, 946, 2739, 979)" in output


def test_cli_help_includes_crop_montage_options(monkeypatch, capsys) -> None:
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
