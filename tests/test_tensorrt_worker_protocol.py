from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_cpp(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_cpp_worker_protocol_is_jsonl_only_on_stdout() -> None:
    source = read_cpp("cpp_inference/src/tensorrt_worker.cpp")

    assert "std::getline(std::cin, line)" in source
    assert "std::cout << json << '\\n' << std::flush" in source
    assert 'std::cerr << "TensorRT worker started' in source
    assert '"event\\":\\"ready' in source
    assert '"status\\":\\"shutdown' in source


def test_cpp_worker_validates_requests_and_keeps_processing() -> None:
    source = read_cpp("cpp_inference/src/tensorrt_worker.cpp")

    for field in ("request_id", "command", "image", "confidence", "iou"):
        assert f'"{field}"' in source
    assert 'request.command != "infer"' in source
    assert 'value < 0.0 || value > 1.0' in source
    assert 'write_error(fallback_request_id' in source
    assert "while (std::getline(std::cin, line))" in source


def test_worker_reuses_one_tensorrt_detector_and_unicode_loader() -> None:
    source = read_cpp("cpp_inference/src/tensorrt_worker.cpp")

    detector_index = source.index("TensorRtDetector detector(")
    loop_index = source.index("while (std::getline(std::cin, line))")
    assert detector_index < loop_index
    assert "load_bgr_image(image_path)" in source
    assert "path_from_utf8(request.image)" in source
    assert "detector.infer(image, request.confidence, request.iou)" in source
