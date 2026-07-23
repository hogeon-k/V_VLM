#pragma once

#include <opencv2/core.hpp>

#include <string>

#include "inference_result.hpp"

namespace pcb_vision {

class Detector {
public:
    virtual ~Detector() = default;

    virtual bool load(const std::string& model_path) = 0;

    virtual InferenceResult infer(
        const cv::Mat& image,
        float confidence_threshold,
        float nms_iou_threshold
    ) = 0;
};

class EnvironmentCheckDetector final : public Detector {
public:
    bool load(const std::string& model_path) override;

    InferenceResult infer(
        const cv::Mat& image,
        float confidence_threshold,
        float nms_iou_threshold
    ) override;

private:
    std::string model_path_;
};

}  // namespace pcb_vision
