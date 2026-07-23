#include "detector.hpp"

#include <stdexcept>

namespace pcb_vision {

bool EnvironmentCheckDetector::load(const std::string& model_path) {
    model_path_ = model_path;
    return !model_path_.empty();
}

InferenceResult EnvironmentCheckDetector::infer(
    const cv::Mat& image,
    float confidence_threshold,
    float nms_iou_threshold
) {
    (void)confidence_threshold;
    (void)nms_iou_threshold;

    if (image.empty()) {
        throw std::invalid_argument("Cannot run inference on an empty image.");
    }

    InferenceResult result;
    result.is_ng = false;
    return result;
}

}  // namespace pcb_vision
