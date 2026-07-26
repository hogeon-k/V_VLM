#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

namespace pcb_vision {

struct LetterboxResult {
    cv::Mat image;
    float scale = 1.0F;
    float pad_x = 0.0F;
    float pad_y = 0.0F;
    int pad_left = 0;
    int pad_right = 0;
    int pad_top = 0;
    int pad_bottom = 0;
    int resized_width = 0;
    int resized_height = 0;
    int original_width = 0;
    int original_height = 0;
};

struct PreprocessResult {
    std::vector<float> tensor;
    std::array<int64_t, 4> shape{};
    LetterboxResult letterbox;
    float min_value = 0.0F;
    float max_value = 0.0F;
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

PreprocessResult preprocess_image(const cv::Mat& bgr_image, int image_size);

}  // namespace pcb_vision
