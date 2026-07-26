#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace pcb_vision {

void configure_utf8_console();

std::vector<std::string> command_line_to_utf8_args(int argc, char* argv[]);

std::wstring utf8_to_wide(const std::string& value);

std::string wide_to_utf8(const std::wstring& value);

std::filesystem::path path_from_utf8(const std::string& value);

std::string path_to_utf8(const std::filesystem::path& path);

std::string read_text_file_utf8(const std::filesystem::path& path);

std::vector<unsigned char> read_binary_file(const std::filesystem::path& path);

void write_utf8_text_file(const std::filesystem::path& path, const std::string& text);

cv::Mat read_image_unicode(const std::filesystem::path& path, int flags = cv::IMREAD_COLOR);

bool write_image_unicode(const std::filesystem::path& path, const cv::Mat& image);

}  // namespace pcb_vision
