#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "image_preprocessor.hpp"
#include "inference_result.hpp"

namespace pcb_vision {

struct TensorRtTensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype;
};

struct TensorRtRunTiming {
    double h2d_ms = 0.0;
    double gpu_execution_ms = 0.0;
    double d2h_ms = 0.0;
    double total_ms = 0.0;
};

class TensorRtDetector final {
public:
    TensorRtDetector(
        std::string engine_path,
        std::vector<std::string> class_names,
        int image_size = 960,
        int device_id = 0
    );

    ~TensorRtDetector();

    TensorRtDetector(const TensorRtDetector&) = delete;
    TensorRtDetector& operator=(const TensorRtDetector&) = delete;
    TensorRtDetector(TensorRtDetector&&) noexcept;
    TensorRtDetector& operator=(TensorRtDetector&&) noexcept;

    InferenceResult infer(
        const cv::Mat& image,
        float confidence_threshold,
        float nms_iou_threshold
    );

    RawInferenceResult run_preprocessed(PreprocessResult& preprocess);

    InferenceResult infer_preprocessed(
        PreprocessResult& preprocess,
        cv::Size original_size,
        float confidence_threshold,
        float nms_iou_threshold
    );

    const TensorRtTensorInfo& input_info() const;
    const TensorRtTensorInfo& output_info() const;
    const TensorRtRunTiming& last_timing() const;
    const std::string& engine_path() const;
    int device_id() const;
    std::string version_string() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pcb_vision
