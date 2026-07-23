#include "image_preprocessor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace pcb_vision {

bool can_load_image(const std::string& image_path) {
    return !cv::imread(image_path, cv::IMREAD_COLOR).empty();
}

cv::Mat load_bgr_image(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    return image;
}

LetterboxResult letterbox_resize(
    const cv::Mat& image,
    int target_width,
    int target_height,
    const cv::Scalar& padding_color
) {
    if (image.empty()) {
        throw std::invalid_argument("letterbox_resize received an empty image.");
    }
    if (target_width <= 0 || target_height <= 0) {
        throw std::invalid_argument("letterbox target size must be positive.");
    }

    const double scale = std::min(
        static_cast<double>(target_width) / static_cast<double>(image.cols),
        static_cast<double>(target_height) / static_cast<double>(image.rows)
    );
    const int resized_width = static_cast<int>(std::round(static_cast<double>(image.cols) * scale));
    const int resized_height = static_cast<int>(std::round(static_cast<double>(image.rows) * scale));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_width, resized_height));

    const int pad_x = (target_width - resized_width) / 2;
    const int pad_y = (target_height - resized_height) / 2;
    cv::Mat canvas(target_height, target_width, image.type(), padding_color);
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, resized_width, resized_height)));

    LetterboxResult result;
    result.image = canvas;
    result.scale = scale;
    result.pad_x = pad_x;
    result.pad_y = pad_y;
    return result;
}

cv::Mat normalize_to_chw_float(const cv::Mat& bgr_image) {
    if (bgr_image.empty()) {
        throw std::invalid_argument("normalize_to_chw_float received an empty image.");
    }

    cv::Mat float_image;
    bgr_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(float_image, channels);
    cv::Mat chw;
    cv::vconcat(channels, chw);
    return chw;
}

}  // namespace pcb_vision
