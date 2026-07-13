from __future__ import annotations

import sys

import scripts.test_yolo_vlm as cli


def test_cli_vlm_defaults_and_skip_vlm_flag(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["test_yolo_vlm.py", "--image", "sample.jpg", "--skip-vlm"])

    args = cli.parse_args()

    assert args.ollama_host == "http://127.0.0.1:11434"
    assert args.vlm_num_ctx == 8192
    assert args.vlm_num_predict == 512
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
        ],
    )

    args = cli.parse_args()
    service = cli.build_vlm_service(args)

    assert service.client.host == "http://example.test:11434"
    assert service.client.num_ctx == 10000
    assert service.client.num_predict == 256
