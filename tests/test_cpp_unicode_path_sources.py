from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_cpp(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_windows_command_line_is_read_as_unicode_and_console_uses_utf8() -> None:
    header = read_cpp("cpp_inference/include/unicode_utils.hpp")
    source = read_cpp("cpp_inference/src/unicode_utils.cpp")
    cmake = read_cpp("cpp_inference/CMakeLists.txt")

    assert "command_line_to_utf8_args" in header
    assert "CommandLineToArgvW(GetCommandLineW()" in source
    assert "WideCharToMultiByte(CP_UTF8" in source
    assert "MultiByteToWideChar(CP_UTF8" in source
    assert "SetConsoleOutputCP(CP_UTF8)" in source
    assert "SetConsoleCP(CP_UTF8)" in source
    assert "shell32" in cmake


def test_cpp_image_loader_uses_binary_decode_for_unicode_paths() -> None:
    header = read_cpp("cpp_inference/include/image_preprocessor.hpp")
    image_source = read_cpp("cpp_inference/src/image_preprocessor.cpp")
    unicode_source = read_cpp("cpp_inference/src/unicode_utils.cpp")

    assert "load_bgr_image(const std::filesystem::path& image_path)" in header
    assert "can_load_image(const std::filesystem::path& image_path)" in header
    assert "read_image_unicode(image_path, cv::IMREAD_COLOR)" in image_source
    assert "cv::imread" not in image_source
    assert "cv::imdecode" in unicode_source
    assert "read_binary_file(const std::filesystem::path& path)" in unicode_source


def test_result_image_and_json_outputs_are_unicode_safe_utf8_files() -> None:
    main_source = read_cpp("cpp_inference/src/main.cpp")
    batch_source = read_cpp("cpp_inference/src/batch_benchmark_main.cpp")
    trt_batch_source = read_cpp("cpp_inference/src/tensorrt_batch_benchmark_main.cpp")
    unicode_source = read_cpp("cpp_inference/src/unicode_utils.cpp")

    assert "write_image_unicode(path, image)" in main_source
    assert "cv::imwrite" not in main_source
    assert "write_utf8_text_file" in unicode_source
    assert "std::ofstream out(path, std::ios::binary)" in main_source
    assert "std::ofstream out(path, std::ios::binary)" in batch_source
    assert "std::ofstream out(path, std::ios::binary)" in trt_batch_source
    assert "path_to_utf8(report.image_path)" in batch_source
    assert "path_to_utf8(report.image_path)" in trt_batch_source


def test_all_cpp_cli_entry_points_convert_argv_before_parsing() -> None:
    for path in [
        "cpp_inference/src/main.cpp",
        "cpp_inference/src/batch_benchmark_main.cpp",
        "cpp_inference/src/tensorrt_batch_benchmark_main.cpp",
    ]:
        source = read_cpp(path)

        assert "configure_utf8_console();" in source
        assert "command_line_to_utf8_args(argc, argv)" in source
        assert "parse_args(utf8_args)" in source
        assert "parse_args(argc, argv)" not in source
        assert "require_value(index, argc, argv" not in source


def test_model_engine_and_image_paths_are_converted_at_filesystem_boundary() -> None:
    detector_source = read_cpp("cpp_inference/src/detector.cpp")
    tensorrt_source = read_cpp("cpp_inference/src/tensorrt_detector.cpp")
    main_source = read_cpp("cpp_inference/src/main.cpp")
    batch_source = read_cpp("cpp_inference/src/batch_benchmark_main.cpp")
    trt_batch_source = read_cpp("cpp_inference/src/tensorrt_batch_benchmark_main.cpp")

    assert "return utf8_to_wide(path);" in detector_source
    assert "read_binary_file(path_from_utf8(path))" in tensorrt_source
    assert "load_bgr_image(pcb_vision::path_from_utf8(args.image))" in main_source
    assert "const fs::path output_root = pcb_vision::path_from_utf8(args.output)" in main_source
    assert "collect_images(pcb_vision::path_from_utf8(args.images)" in batch_source
    assert "collect_images(pcb_vision::path_from_utf8(args.images)" in trt_batch_source
