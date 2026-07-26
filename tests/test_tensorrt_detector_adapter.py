from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

import scripts.test_yolo_vlm as cli
from service import tensorrt_detector_adapter as adapter_module
from service.tensorrt_detector_adapter import (
    TensorRtAdapterError,
    TensorRtDetectorAdapter,
    TensorRtExecutionError,
    TensorRtResultParseError,
)


def write_image(path: Path) -> None:
    Image.new("RGB", (100, 80), color=(255, 255, 255)).save(path)


def make_adapter_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    executable = tmp_path / "pcb_onnx_infer.exe"
    engine = tmp_path / "best_fp16.engine"
    metadata = tmp_path / "model_metadata.json"
    image = tmp_path / "sample.jpg"
    executable.write_text("fake exe", encoding="utf-8")
    engine.write_bytes(b"engine")
    metadata.write_text('{"class_names": ["open_circuit", "short", "missing_hole"]}', encoding="utf-8")
    write_image(image)
    return executable, engine, metadata, image


def make_adapter(tmp_path: Path, **kwargs: object) -> TensorRtDetectorAdapter:
    executable, engine, metadata, _ = make_adapter_files(tmp_path)
    return TensorRtDetectorAdapter(
        executable,
        engine,
        metadata,
        device_id=1,
        image_size=960,
        confidence_threshold=0.15,
        iou_threshold=0.7,
        engine_label="fp16",
        **kwargs,
    )


def fake_success_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    output_dir = Path(command[command.index("--output") + 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        """
        {
          "backend": "tensorrt",
          "engine_label": "fp16",
          "timing_ms": {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0, "total": 6.0},
          "detections": [
            {"class_id": 2, "class_name": "missing_hole", "confidence": 0.911177, "bbox": [10.2, 20.4, 30.6, 40.8]},
            {"class_id": 2, "class_name": "missing_hole", "confidence": 0.894052, "bbox": [50, 20, 70, 45]}
          ]
        }
        """,
        encoding="utf-8",
    )
    write_image(output_dir / "result.jpg")
    return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")


def test_tensorrt_adapter_builds_expected_command(monkeypatch, tmp_path) -> None:
    _, _, _, image = make_adapter_files(tmp_path)
    detector = make_adapter(tmp_path)
    commands: list[list[str]] = []

    def capture(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return fake_success_run(command, **kwargs)

    monkeypatch.setattr(adapter_module.subprocess, "run", capture)

    detector.detect(image)

    command = commands[0]
    assert command[:3] == [str(detector.executable_path), "--backend", "tensorrt"]
    assert command[command.index("--engine") + 1] == str(detector.engine_path)
    assert command[command.index("--metadata") + 1] == str(detector.metadata_path)
    assert command[command.index("--image") + 1] == str(image.resolve())
    assert command[command.index("--engine-label") + 1] == "fp16"
    assert command[command.index("--device-id") + 1] == "1"
    assert command[command.index("--conf") + 1] == "0.15"
    assert command[command.index("--iou") + 1] == "0.7"
    assert command[command.index("--warmup") + 1] == "0"
    assert command[command.index("--repeat") + 1] == "1"


def test_tensorrt_adapter_converts_json_to_yolo_result(monkeypatch, tmp_path) -> None:
    _, _, _, image = make_adapter_files(tmp_path)
    detector = make_adapter(tmp_path)
    monkeypatch.setattr(adapter_module.subprocess, "run", fake_success_run)

    result = detector.detect(image)

    assert result.image_path == image.resolve()
    assert result.is_ng is True
    assert result.defect_count == 2
    assert result.annotated_image_path is not None
    assert result.annotated_image_path.is_file()
    first = result.detections[0]
    assert first.class_id == 2
    assert first.class_name == "missing_hole"
    assert first.confidence == pytest.approx(0.911177)
    assert (first.x1, first.y1, first.x2, first.y2) == (10, 20, 31, 41)
    assert first.location is not None
    assert detector.last_metadata is not None
    assert detector.last_metadata.detection_count == 2
    assert detector.last_metadata.timing_ms["inference"] == 2.0


def test_tensorrt_adapter_handles_zero_detections(monkeypatch, tmp_path) -> None:
    _, _, _, image = make_adapter_files(tmp_path)
    detector = make_adapter(tmp_path)

    def run_zero(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text('{"backend": "tensorrt", "detections": []}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(adapter_module.subprocess, "run", run_zero)

    result = detector.detect(image)

    assert result.is_ng is False
    assert result.detections == []


def test_tensorrt_adapter_rejects_missing_required_files(tmp_path) -> None:
    executable, engine, metadata, _ = make_adapter_files(tmp_path)

    with pytest.raises(TensorRtAdapterError, match="TensorRT engine not found"):
        TensorRtDetectorAdapter(executable, engine.with_name("missing.engine"), metadata)


def test_tensorrt_adapter_reports_nonzero_exit(monkeypatch, tmp_path) -> None:
    _, _, _, image = make_adapter_files(tmp_path)
    detector = make_adapter(tmp_path)

    def fail(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, stdout="out", stderr="dll load failed")

    monkeypatch.setattr(adapter_module.subprocess, "run", fail)

    with pytest.raises(TensorRtExecutionError) as exc_info:
        detector.detect(image)

    message = str(exc_info.value)
    assert "returncode=2" in message
    assert "dll load failed" in message
    assert str(detector.engine_path) in message


def test_tensorrt_adapter_reports_timeout(monkeypatch, tmp_path) -> None:
    _, _, _, image = make_adapter_files(tmp_path)
    detector = make_adapter(tmp_path, timeout_seconds=0.01)

    def timeout(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 0.01, output="partial", stderr="still running")

    monkeypatch.setattr(adapter_module.subprocess, "run", timeout)

    with pytest.raises(TensorRtExecutionError, match="timed out"):
        detector.detect(image)


def test_tensorrt_adapter_reports_invalid_result_schema(monkeypatch, tmp_path) -> None:
    _, _, _, image = make_adapter_files(tmp_path)
    detector = make_adapter(tmp_path)

    def invalid(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text('{"detections": [{"class_id": 99, "confidence": 0.1, "bbox": [1, 2, 3, 4]}]}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(adapter_module.subprocess, "run", invalid)

    with pytest.raises(TensorRtResultParseError, match="class_id out of range"):
        detector.detect(image)


def test_tensorrt_adapter_cleans_temporary_output_directory(monkeypatch, tmp_path) -> None:
    _, _, _, image = make_adapter_files(tmp_path)
    detector = make_adapter(tmp_path)
    output_dirs: list[Path] = []

    def capture_output_dir(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output_dirs.append(Path(command[command.index("--output") + 1]))
        return fake_success_run(command, **kwargs)

    monkeypatch.setattr(adapter_module.subprocess, "run", capture_output_dir)

    detector.detect(image)

    assert output_dirs
    assert not output_dirs[0].exists()


def test_cli_backend_defaults_to_pytorch(monkeypatch) -> None:
    monkeypatch.setattr(cli.sys, "argv", ["test_yolo_vlm.py", "--image", "sample.jpg"])

    args = cli.parse_args()

    assert args.backend == "pytorch"


def test_cli_factory_selects_tensorrt(monkeypatch, tmp_path) -> None:
    executable, engine, metadata, _ = make_adapter_files(tmp_path)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "test_yolo_vlm.py",
            "--image",
            "sample.jpg",
            "--backend",
            "tensorrt",
            "--tensorrt-executable",
            str(executable),
            "--tensorrt-engine",
            str(engine),
            "--tensorrt-metadata",
            str(metadata),
            "--device-id",
            "2",
        ],
    )

    service = cli.build_yolo_service(cli.parse_args())

    assert isinstance(service.detector, TensorRtDetectorAdapter)
    assert service.detector.device_id == 2

