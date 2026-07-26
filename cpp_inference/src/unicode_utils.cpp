#include "unicode_utils.hpp"

#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#endif

namespace pcb_vision {

namespace {

std::string extension_for_imencode(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    return extension.empty() ? ".jpg" : extension;
}

}  // namespace

void configure_utf8_console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

std::wstring utf8_to_wide(const std::string& value) {
#ifdef _WIN32
    if (value.empty()) {
        return {};
    }
    const int wide_size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0);
    if (wide_size <= 0) {
        throw std::runtime_error("Failed to convert UTF-8 text to UTF-16.");
    }
    std::wstring wide(static_cast<std::size_t>(wide_size), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), wide.data(), wide_size);
    return wide;
#else
    return std::wstring(value.begin(), value.end());
#endif
}

std::string wide_to_utf8(const std::wstring& value) {
#ifdef _WIN32
    if (value.empty()) {
        return {};
    }
    const int utf8_size = WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (utf8_size <= 0) {
        throw std::runtime_error("Failed to convert UTF-16 text to UTF-8.");
    }
    std::string utf8(static_cast<std::size_t>(utf8_size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), utf8.data(), utf8_size, nullptr, nullptr);
    return utf8;
#else
    return std::string(value.begin(), value.end());
#endif
}

std::vector<std::string> command_line_to_utf8_args(int argc, char* argv[]) {
#ifdef _WIN32
    (void)argc;
    (void)argv;
    int wide_argc = 0;
    LPWSTR* wide_argv = CommandLineToArgvW(GetCommandLineW(), &wide_argc);
    if (wide_argv == nullptr) {
        throw std::runtime_error("Failed to read Windows Unicode command line.");
    }
    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(wide_argc));
    for (int index = 0; index < wide_argc; ++index) {
        args.push_back(wide_to_utf8(wide_argv[index]));
    }
    LocalFree(wide_argv);
    return args;
#else
    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(argc));
    for (int index = 0; index < argc; ++index) {
        args.emplace_back(argv[index]);
    }
    return args;
#endif
}

std::filesystem::path path_from_utf8(const std::string& value) {
#ifdef _WIN32
    return std::filesystem::path(utf8_to_wide(value));
#else
    return std::filesystem::path(value);
#endif
}

std::string path_to_utf8(const std::filesystem::path& path) {
#ifdef _WIN32
    return wide_to_utf8(path.wstring());
#else
    const auto value = path.u8string();
    return std::string(value.begin(), value.end());
#endif
}

std::vector<unsigned char> read_binary_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open file: " + path_to_utf8(path));
    }
    return std::vector<unsigned char>(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()
    );
}

std::string read_text_file_utf8(const std::filesystem::path& path) {
    const std::vector<unsigned char> bytes = read_binary_file(path);
    return std::string(bytes.begin(), bytes.end());
}

void write_utf8_text_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to write file: " + path_to_utf8(path));
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

cv::Mat read_image_unicode(const std::filesystem::path& path, int flags) {
    const std::vector<unsigned char> bytes = read_binary_file(path);
    return cv::imdecode(bytes, flags);
}

bool write_image_unicode(const std::filesystem::path& path, const cv::Mat& image) {
    std::vector<unsigned char> buffer;
    if (!cv::imencode(extension_for_imencode(path), image, buffer)) {
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    return static_cast<bool>(out);
}

}  // namespace pcb_vision
