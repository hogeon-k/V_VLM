from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from scripts import compare_all_backends


def required_cli_args(tmp_path: Path) -> list[str]:
    return [
        "--images",
        str(tmp_path / "images"),
        "--pytorch-model",
        str(tmp_path / "best.pt"),
        "--onnx-model",
        str(tmp_path / "best.onnx"),
        "--tensorrt-fp32-engine",
        str(tmp_path / "fp32.engine"),
        "--tensorrt-fp16-engine",
        str(tmp_path / "fp16.engine"),
        "--metadata",
        str(tmp_path / "metadata.json"),
        "--output",
        str(tmp_path / "output"),
    ]


def test_cli_parses_required_paths_and_recommended_defaults(tmp_path) -> None:
    args = compare_all_backends.parse_args(required_cli_args(tmp_path))

    assert args.images == tmp_path / "images"
    assert args.imgsz == 960
    assert args.conf == 0.15
    assert args.iou == 0.5
    assert args.match_iou == 0.5
    assert args.warmup == 5
    assert args.repeat == 20
    assert args.device == "0"
    assert args.provider == "CUDAExecutionProvider"


def test_cli_parses_extensions_limit_and_fail_policy(tmp_path) -> None:
    args = compare_all_backends.parse_args(
        [
            *required_cli_args(tmp_path),
            "--extensions",
            ".jpg,.png",
            "--max-images",
            "3",
            "--fail-on-mismatch",
        ]
    )

    assert args.extensions == ".jpg,.png"
    assert args.max_images == 3
    assert args.fail_on_mismatch is True


def test_main_returns_benchmark_exit_code(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        compare_all_backends,
        "run_benchmark",
        lambda args: SimpleNamespace(
            summary={"final_status": "FAIL"},
            exit_code=1,
        ),
    )

    assert compare_all_backends.main(required_cli_args(tmp_path)) == 1


def test_main_returns_two_for_validation_error(monkeypatch, tmp_path) -> None:
    def fail(args: Namespace) -> object:
        raise ValueError("bad arguments")

    monkeypatch.setattr(compare_all_backends, "run_benchmark", fail)

    assert compare_all_backends.main(required_cli_args(tmp_path)) == 2
