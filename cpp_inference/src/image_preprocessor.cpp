#include "image_preprocessor.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

#include "unicode_utils.hpp"

namespace pcb_vision {
namespace {

int round_half_to_even(double value) {
    const double lower_value = std::floor(value);
    const double fraction = value - lower_value;
    const int lower = static_cast<int>(lower_value);
    if (fraction < 0.5) {
        return lower;
    }
    if (fraction > 0.5) {
        return lower + 1;
    }
    return lower % 2 == 0 ? lower : lower + 1;
}

}  // namespace

bool can_load_image(const std::filesystem::path& image_path) {
    return !read_image_unicode(image_path, cv::IMREAD_COLOR).empty();
}

cv::Mat load_bgr_image(const std::filesystem::path& image_path) {
    cv::Mat image = read_image_unicode(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + path_to_utf8(image_path));
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
    const int resized_width = round_half_to_even(static_cast<double>(image.cols) * scale);
    const int resized_height = round_half_to_even(static_cast<double>(image.rows) * scale);
    if (resized_width <= 0 || resized_height <= 0) {
        throw std::runtime_error("letterbox resize produced a non-positive image size.");
    }
    if (resized_width > target_width || resized_height > target_height) {
        throw std::runtime_error("letterbox resize exceeds the target size.");
    }

    cv::Mat resized;
    if (image.cols != resized_width || image.rows != resized_height) {
        cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        resized = image;
    }

    float dw = static_cast<float>(target_width - resized_width) / 2.0F;
    float dh = static_cast<float>(target_height - resized_height) / 2.0F;
    const int left = static_cast<int>(std::round(dw - 0.1F));
    const int right = static_cast<int>(std::round(dw + 0.1F));
    const int top = static_cast<int>(std::round(dh - 0.1F));
    const int bottom = static_cast<int>(std::round(dh + 0.1F));

    cv::Mat canvas;
    cv::copyMakeBorder(resized, canvas, top, bottom, left, right, cv::BORDER_CONSTANT, padding_color);

    LetterboxResult result;
    result.image = canvas;
    result.scale = static_cast<float>(scale);
    result.pad_x = static_cast<float>(left);
    result.pad_y = static_cast<float>(top);
    result.pad_left = left;
    result.pad_right = right;
    result.pad_top = top;
    result.pad_bottom = bottom;
    result.resized_width = resized_width;
    result.resized_height = resized_height;
    result.original_width = image.cols;
    result.original_height = image.rows;
    return result;
}

cv::Mat normalize_to_chw_float(const cv::Mat& bgr_image) {
    if (bgr_image.empty()) {
        throw std::invalid_argument("normalize_to_chw_float received an empty image.");
    }

    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(float_image, channels);
    cv::Mat chw;
    cv::vconcat(channels, chw);
    return chw;
}

PreprocessResult preprocess_image(const cv::Mat& bgr_image, int image_size) {
    LetterboxResult letterbox = letterbox_resize(
        bgr_image,
        image_size,
        image_size,
        cv::Scalar(114, 114, 114)
    );

    cv::Mat rgb_image;
    cv::cvtColor(letterbox.image, rgb_image, cv::COLOR_BGR2RGB);

    PreprocessResult result;
    result.shape = {1, 3, image_size, image_size};
    result.letterbox = letterbox;
    result.tensor.resize(static_cast<std::size_t>(3 * image_size * image_size));
    result.min_value = std::numeric_limits<float>::max();
    result.max_value = std::numeric_limits<float>::lowest();

    const int channel_stride = image_size * image_size;
    for (int y = 0; y < image_size; ++y) {
        const cv::Vec3b* row = rgb_image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < image_size; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                const float value = static_cast<float>(row[x][channel]) / 255.0F;
                result.tensor[static_cast<std::size_t>(channel * channel_stride + y * image_size + x)] = value;
                result.min_value = std::min(result.min_value, value);
                result.max_value = std::max(result.max_value, value);
            }
        }
    }
    return result;
}

}  // namespace pcb_vision
