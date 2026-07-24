#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "inference_result.hpp"

namespace pcb_vision {

class OnnxDetector final {
public:
    OnnxDetector(
        std::string model_path,
        std::vector<std::string> class_names,
        int image_size = 960
    );
    ~OnnxDetector();

    InferenceResult infer(
        const cv::Mat& image,
        float confidence_threshold,
        float nms_iou_threshold
    );

    const std::vector<int64_t>& input_shape() const;
    const std::vector<int64_t>& output_shape() const;
    const std::string& input_name() const;
    const std::string& output_name() const;
    const std::string& provider() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string model_path_;
    std::vector<std::string> class_names_;
    int image_size_ = 960;
};

}  // namespace pcb_vision
