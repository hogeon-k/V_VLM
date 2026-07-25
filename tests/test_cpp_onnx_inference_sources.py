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
    assert "onnxruntime_providers_cuda.dll" in cmake
    assert "onnxruntime_providers_shared.dll" in cmake


def test_cpp_cuda_provider_uses_v2_options_and_reports_cudnn_search() -> None:
    source = read_cpp("cpp_inference/src/detector.cpp")
    main = read_cpp("cpp_inference/src/main.cpp")

    assert "CreateCUDAProviderOptions" in source
    assert "UpdateCUDAProviderOptions" in source
    assert "AppendExecutionProvider_CUDA_V2" in source
    assert "\"device_id\", \"cudnn_conv_algo_search\"" in source
    assert "cudnn_conv_algo_search" in main
    assert "--cudnn-conv-algo-search" in main


def test_cpp_benchmark_warmup_repeat_and_session_run_stats_are_present() -> None:
    source = read_cpp("cpp_inference/src/main.cpp")
    detector_header = read_cpp("cpp_inference/include/detector.hpp")

    assert "--warmup" in source
    assert "--repeat" in source
    assert "Warmup: " in source
    assert "Session.Run stats (ms)" in source
    assert "benchmark.json" in source
    assert "benchmark.csv" in source
    assert "run_preprocessed" in detector_header
    assert "infer_preprocessed" in detector_header


def test_cpp_benchmark_validates_detection_stability() -> None:
    source = read_cpp("cpp_inference/src/main.cpp")

    assert "detection count mismatch" in source
    assert "class_id mismatch" in source
    assert "confidence mismatch" in source
    assert "> 0.001F" in source
    assert "> 1.0F" in source
    assert "Validation mismatches" in source


def test_cpp_cpu_cuda_batch_benchmark_executable_and_outputs_are_present() -> None:
    cmake = read_cpp("cpp_inference/CMakeLists.txt")
    source = read_cpp("cpp_inference/src/batch_benchmark_main.cpp")
    utilities = read_cpp("cpp_inference/src/batch_benchmark.cpp")

    assert "pcb_onnx_batch_benchmark" in cmake
    assert "--images" in source
    assert "--provider-order" in source
    assert "--confidence-tolerance" in source
    assert "--bbox-tolerance" in source
    assert "summary.json" in source
    assert "image_results.csv" in source
    assert "timing_runs.csv" in source
    assert "environment.json" in source
    assert "failure_cases" in source
    assert "compare_detections" in utilities
    assert "collect_images" in utilities


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
