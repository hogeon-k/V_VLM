#include "postprocessor.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void run_preprocessing_tests();

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_iou() {
    const cv::Rect2f first(0.0F, 0.0F, 10.0F, 10.0F);
    const cv::Rect2f second(5.0F, 5.0F, 10.0F, 10.0F);

    require(std::abs(pcb_vision::bbox_iou(first, first) - 1.0F) < 1e-6F, "self IoU mismatch");
    require(std::abs(pcb_vision::bbox_iou(first, second) - (25.0F / 175.0F)) < 1e-6F, "overlap IoU mismatch");
}

void test_restore_box_float_coordinates() {
    pcb_vision::LetterboxResult letterbox;
    letterbox.scale = 4.8F;
    letterbox.pad_x = 0.0F;
    letterbox.pad_y = 240.0F;

    const cv::Rect2f restored = pcb_vision::restore_box_to_original_image(
        cv::Rect2f(240.0F, 336.0F, 480.0F, 288.0F),
        letterbox,
        cv::Size(200, 100)
    );

    require(std::abs(restored.x - 50.0F) < 1e-5F, "restore x mismatch");
    require(std::abs(restored.y - 20.0F) < 1e-5F, "restore y mismatch");
    require(std::abs(restored.width - 100.0F) < 1e-5F, "restore width mismatch");
    require(std::abs(restored.height - 60.0F) < 1e-5F, "restore height mismatch");
}

void test_decode_and_class_aware_nms() {
    const std::vector<std::string> class_names = {"open_circuit", "short", "missing_hole"};
    pcb_vision::LetterboxResult letterbox;
    letterbox.scale = 1.0F;
    letterbox.pad_x = 0.0F;
    letterbox.pad_y = 0.0F;

    // Shape [1, 7, 3], laid out as [channel][candidate].
    // Candidate 1 is centered at (10.5, 10.5) so its IoU with candidate 0 is
    // safely above the 0.7 NMS threshold; (11, 11) is only about 0.68.
    std::vector<float> output = {
        10.0F, 10.5F, 10.0F,  // x center
        10.0F, 10.5F, 10.0F,  // y center
        10.0F, 10.0F, 10.0F,  // width
        10.0F, 10.0F, 10.0F,  // height
        0.9F, 0.8F, 0.1F,     // class 0
        0.1F, 0.2F, 0.85F,    // class 1
        0.0F, 0.0F, 0.05F     // class 2
    };

    const std::vector<pcb_vision::Detection> detections = pcb_vision::decode_yolo_output(
        output.data(),
        {1, 7, 3},
        letterbox,
        cv::Size(100, 100),
        0.15F,
        0.7F,
        class_names
    );

    require(detections.size() == 2, "class-aware NMS should keep one class 0 and one class 1 box");
    require(detections[0].class_id == 0, "highest confidence class mismatch");
    require(detections[1].class_id == 1, "different class box should not be suppressed");
}

void test_invalid_shape() {
    const std::vector<std::string> class_names = {"open_circuit", "short", "missing_hole"};
    pcb_vision::LetterboxResult letterbox;
    std::vector<float> output(7, 0.0F);
    bool threw = false;
    try {
        (void)pcb_vision::decode_yolo_output(output.data(), {7, 1}, letterbox, cv::Size(10, 10), 0.1F, 0.7F, class_names);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "invalid output shape should throw");
}

}  // namespace

int main() {
    try {
        run_preprocessing_tests();
        test_iou();
        test_restore_box_float_coordinates();
        test_decode_and_class_aware_nms();
        test_invalid_shape();
        std::cout << "postprocessing tests passed\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "test failed: " << exc.what() << '\n';
        return 1;
    }
}
