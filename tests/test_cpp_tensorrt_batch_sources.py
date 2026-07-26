from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_cpp(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_tensorrt_batch_target_is_wired_to_shared_sources() -> None:
    cmake = read_cpp("cpp_inference/CMakeLists.txt")

    assert "pcb_tensorrt_batch_benchmark" in cmake
    assert "src/tensorrt_batch_benchmark_main.cpp" in cmake
    assert "src/batch_benchmark.cpp" in cmake
    assert "src/tensorrt_detector.cpp" in cmake
    assert "src/image_preprocessor.cpp" in cmake
    assert "src/postprocessor.cpp" in cmake
    assert "TENSORRT_NVINFER_LIBRARY" in cmake
    assert "CUDA::cudart" in cmake
    assert "OpenCV_LIBS" in cmake
    assert "nvonnxparser" not in cmake
    assert "nvinfer_builder_resource" not in cmake


def test_tensorrt_batch_cli_and_outputs_are_present() -> None:
    source = read_cpp("cpp_inference/src/tensorrt_batch_benchmark_main.cpp")

    for option in [
        "--engine",
        "--engine-label",
        "--images",
        "--output",
        "--metadata",
        "--device-id",
        "--imgsz",
        "--conf",
        "--iou",
        "--match-iou",
        "--warmup",
        "--repeat",
        "--strict-confidence-tolerance",
        "--practical-confidence-tolerance",
        "--bbox-tolerance",
    ]:
        assert option in source

    assert "summary.json" in source
    assert "per_image.csv" in source
    assert "detections.json" in source
    assert "failure_cases" in source
    assert ".gitkeep" in source


def test_tensorrt_batch_reuses_detector_and_benchmark_utilities() -> None:
    source = read_cpp("cpp_inference/src/tensorrt_batch_benchmark_main.cpp")

    assert "pcb_vision::TensorRtDetector detector" in source
    assert "detector.infer_preprocessed" in source
    assert "pcb_vision::preprocess_image" in source
    assert "bench::collect_images" in source
    assert "bench::compare_detections" in source
    assert "bench::calculate_stats" in source
    assert "bench::validate_confidence_tolerances" in source
    assert "timings.validation_mismatches" in source
    assert "for (int warmup = 0; warmup < args.warmup; ++warmup)" in source
    assert "for (int repeat = 0; repeat < args.repeat; ++repeat)" in source


def test_tensorrt_batch_uses_native_tensorrt10_path_only() -> None:
    batch_source = read_cpp("cpp_inference/src/tensorrt_batch_benchmark_main.cpp")
    detector_source = read_cpp("cpp_inference/src/tensorrt_detector.cpp")

    assert "enqueueV3" in detector_source
    assert "setTensorAddress" in detector_source
    assert "getNbIOTensors" in detector_source
    assert "getBindingIndex" not in detector_source
    assert "getNbBindings" not in detector_source
    assert "enqueueV2" not in detector_source
    assert "OnnxDetector" not in batch_source
    assert "ONNX Runtime" not in batch_source
