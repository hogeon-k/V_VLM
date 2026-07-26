from __future__ import annotations

from pathlib import Path

from scripts.validate_onnx_model import build_metadata, has_dynamic_shape, sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.onnx"
    path.write_bytes(b"abc")

    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_dynamic_shape_detection() -> None:
    assert has_dynamic_shape([[1, 3, 960, 960]]) is False
    assert has_dynamic_shape([[1, 3, "height", 960]]) is True
    assert has_dynamic_shape([[1, 3, None, 960]]) is True


def test_build_metadata_preserves_class_order(tmp_path: Path, monkeypatch) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names:\n  1: short\n  0: open_circuit\n  2: missing_hole\n", encoding="utf-8")
    monkeypatch.setattr("scripts.validate_onnx_model.PROJECT_ROOT", tmp_path)
    validation = {
        "model": "models/best.onnx",
        "inputs": [{"name": "images", "shape": [1, 3, 960, 960]}],
        "outputs": [{"name": "output0", "shape": [1, 7, 18900]}],
        "dynamic_shape": False,
        "opset": 12,
        "file_size_bytes": 123,
        "sha256": "abc",
    }

    metadata = build_metadata(validation, Path("models/best.pt"), data_yaml)

    assert metadata["class_names"] == ["open_circuit", "short", "missing_hole"]
    assert metadata["input_size"] == [960, 960]
    assert metadata["output_names"] == ["output0"]
