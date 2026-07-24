from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_python_cpp_onnx import bbox_iou, match_detections


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_cpp(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_cpp_preprocessing_matches_python_letterbox_contract() -> None:
    source = read_cpp("cpp_inference/src/image_preprocessor.cpp")

    assert "std::round(dw - 0.1F)" in source
    assert "std::round(dh + 0.1F)" in source
    assert "cv::COLOR_BGR2RGB" in source
    assert "1.0 / 255.0" in source
    assert "channel * channel_stride" in source


def test_cpp_postprocessor_decodes_channel_candidate_layout_without_extra_sigmoid() -> None:
    source = read_cpp("cpp_inference/src/postprocessor.cpp")

    assert "output_data[(4 + class_offset) * candidate_count + candidate_index]" in source
    assert "best_score < confidence_threshold" in source
    assert "restore_box_to_original_image" in source
    assert "std::map<int, std::vector<Candidate>> by_class" in source
    assert "sigmoid" not in source.lower()
    assert "objectness" not in source.lower()


def test_cpp_cmake_requires_onnxruntime_root() -> None:
    cmake = read_cpp("cpp_inference/CMakeLists.txt")

    assert "ONNXRUNTIME_ROOT" in cmake
    assert "onnxruntime_cxx_api.h" in cmake
    assert "onnxruntime.dll" in cmake


def test_python_cpp_matcher_passes_equal_detections() -> None:
    py = [{"class_id": 1, "class_name": "short", "confidence": 0.9, "bbox": [0.0, 0.0, 10.0, 10.0]}]
    cpp = [{"class_id": 1, "class_name": "short", "confidence": 0.899, "bbox": [0.0, 0.0, 10.0, 10.0]}]

    matches = match_detections(py, cpp, match_iou=0.5)

    assert matches[0]["status"] == "MATCHED"
    assert matches[0]["confidence_diff_abs"] < 0.01
    assert bbox_iou(py[0]["bbox"], cpp[0]["bbox"]) == 1.0


def test_model_metadata_class_names_available() -> None:
    metadata = json.loads((PROJECT_ROOT / "models/model_metadata.json").read_text(encoding="utf-8"))

    assert metadata["class_names"] == ["open_circuit", "short", "missing_hole"]
