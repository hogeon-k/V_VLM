#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace pcb_vision {

struct Detection {
    int class_id = -1;
    std::string class_name;
    float confidence = 0.0F;
    cv::Rect box;
};

struct InferenceResult {
    bool is_ng = false;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    std::vector<Detection> detections;
};

}  // namespace pcb_vision
