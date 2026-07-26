from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_cpp(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_main_supports_worker_without_changing_oneshot_entrypoint() -> None:
    main = read_cpp("cpp_inference/src/main.cpp")
    cmake = read_cpp("cpp_inference/CMakeLists.txt")

    assert 'option == "--worker"' in main
    assert "if (args.image.empty() && !args.worker)" in main
    assert "run_tensorrt_worker(" in main
    assert "src/tensorrt_worker.cpp" in cmake
    assert "write_json(output_dir / \"result.json\"" in main


def test_worker_response_preserves_detection_and_timing_schema() -> None:
    source = read_cpp("cpp_inference/src/tensorrt_worker.cpp")

    for field in (
        "class_id",
        "class_name",
        "confidence",
        "bbox",
        "preprocess",
        "inference",
        "postprocess",
        "total",
    ):
        assert field in source
