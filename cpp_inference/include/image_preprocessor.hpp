#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace pcb_vision {

struct LetterboxResult {
    cv::Mat image;
    double scale = 1.0;
    int pad_x = 0;
    int pad_y = 0;
};

bool can_load_image(const std::string& image_path);

cv::Mat load_bgr_image(const std::string& image_path);

LetterboxResult letterbox_resize(
    const cv::Mat& image,
    int target_width,
    int target_height,
    const cv::Scalar& padding_color = cv::Scalar(114, 114, 114)
);

cv::Mat normalize_to_chw_float(const cv::Mat& bgr_image);

}  // namespace pcb_vision
