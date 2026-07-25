#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "inference_result.hpp"
#include "image_preprocessor.hpp"

namespace pcb_vision {

struct CudaProviderConfig {
    int device_id = 0;
    std::string cudnn_conv_algo_search = "EXHAUSTIVE";
};

class OnnxDetector final {
public:
    OnnxDetector(
        std::string model_path,
        std::vector<std::string> class_names,
        int image_size = 960,
        std::string requested_provider = "CUDAExecutionProvider",
        CudaProviderConfig cuda_config = {}
    );
    ~OnnxDetector();

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

    const std::vector<int64_t>& input_shape() const;
    const std::vector<int64_t>& output_shape() const;
    const std::string& input_name() const;
    const std::string& output_name() const;
    const std::string& provider() const;
    const std::vector<std::string>& available_providers() const;
    bool cuda_requested() const;
    bool cuda_registered() const;
    bool cpu_fallback_enabled() const;
    const CudaProviderConfig& cuda_config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string model_path_;
    std::vector<std::string> class_names_;
    int image_size_ = 960;
};

std::vector<std::string> available_execution_providers();

}  // namespace pcb_vision
