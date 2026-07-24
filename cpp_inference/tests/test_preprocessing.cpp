#include "image_preprocessor.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_letterbox_scale_and_padding() {
    const cv::Mat image(100, 200, CV_8UC3, cv::Scalar(10, 20, 30));
    const pcb_vision::LetterboxResult result = pcb_vision::letterbox_resize(image, 960, 960);

    require(result.image.cols == 960 && result.image.rows == 960, "letterbox output shape mismatch");
    require(std::abs(result.scale - 4.8F) < 1e-5F, "letterbox scale mismatch");
    require(result.resized_width == 960 && result.resized_height == 480, "resized shape mismatch");
    require(result.pad_left == 0 && result.pad_right == 0, "x padding mismatch");
    require(result.pad_top == 240 && result.pad_bottom == 240, "y padding mismatch");
}

void test_preprocess_rgb_chw_normalization() {
    cv::Mat image(1, 1, CV_8UC3);
    image.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 127, 255);

    const pcb_vision::PreprocessResult result = pcb_vision::preprocess_image(image, 1);

    require(result.shape == std::array<int64_t, 4>{1, 3, 1, 1}, "preprocess shape mismatch");
    require(result.tensor.size() == 3, "tensor size mismatch");
    require(std::abs(result.tensor[0] - 1.0F) < 1e-6F, "red channel mismatch");
    require(std::abs(result.tensor[1] - (127.0F / 255.0F)) < 1e-6F, "green channel mismatch");
    require(std::abs(result.tensor[2] - 0.0F) < 1e-6F, "blue channel mismatch");
}

}  // namespace

void run_preprocessing_tests() {
    test_letterbox_scale_and_padding();
    test_preprocess_rgb_chw_normalization();
    std::cout << "preprocessing tests passed\n";
}
