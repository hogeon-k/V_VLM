#include "detector.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "image_preprocessor.hpp"
#include "postprocessor.hpp"

namespace pcb_vision {
namespace {

double elapsed_ms(const std::chrono::steady_clock::time_point& start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

#ifdef _WIN32
std::wstring to_ort_path(const std::string& path) {
    return std::wstring(path.begin(), path.end());
}
#endif

void validate_shape(
    const std::vector<int64_t>& actual,
    const std::vector<int64_t>& expected,
    const std::string& label
) {
    if (actual != expected) {
        throw std::runtime_error(label + " shape mismatch.");
    }
}

std::string join_strings(const std::vector<std::string>& values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << values[i];
    }
    return out.str();
}

bool contains_provider(const std::vector<std::string>& providers, const std::string& provider) {
    return std::find(providers.begin(), providers.end(), provider) != providers.end();
}

struct CudaProviderOptionsDeleter {
    void operator()(OrtCUDAProviderOptionsV2* options) const {
        Ort::GetApi().ReleaseCUDAProviderOptions(options);
    }
};

using CudaProviderOptionsPtr = std::unique_ptr<OrtCUDAProviderOptionsV2, CudaProviderOptionsDeleter>;

CudaProviderOptionsPtr make_cuda_provider_options(const CudaProviderConfig& config) {
    const OrtApi& api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* raw_options = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&raw_options));
    CudaProviderOptionsPtr options(raw_options);

    const std::string device_id_text = std::to_string(config.device_id);
    const char* keys[] = {"device_id", "cudnn_conv_algo_search"};
    const char* values[] = {device_id_text.c_str(), config.cudnn_conv_algo_search.c_str()};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(options.get(), keys, values, 2));
    return options;
}

void append_cuda_provider(Ort::SessionOptions& session_options, const CudaProviderConfig& config) {
    CudaProviderOptionsPtr cuda_options = make_cuda_provider_options(config);
    session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
}

}  // namespace

struct OnnxDetector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "pcb_onnx_infer"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    std::string provider = "CPUExecutionProvider";
    std::vector<std::string> available_providers;
    bool cuda_requested = false;
    bool cuda_registered = false;
    bool cpu_fallback_enabled = true;
    CudaProviderConfig cuda_config;
};

OnnxDetector::OnnxDetector(
    std::string model_path,
    std::vector<std::string> class_names,
    int image_size,
    std::string requested_provider,
    CudaProviderConfig cuda_config
) : impl_(std::make_unique<Impl>()),
    model_path_(std::move(model_path)),
    class_names_(std::move(class_names)),
    image_size_(image_size) {
    if (class_names_.empty()) {
        throw std::invalid_argument("class_names must not be empty.");
    }
    if (requested_provider != "CUDAExecutionProvider" && requested_provider != "CPUExecutionProvider") {
        throw std::invalid_argument("requested_provider must be CUDAExecutionProvider or CPUExecutionProvider.");
    }

    impl_->provider = requested_provider;
    impl_->cuda_requested = (requested_provider == "CUDAExecutionProvider");
    impl_->cuda_config = std::move(cuda_config);
    impl_->available_providers = Ort::GetAvailableProviders();
    impl_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (impl_->cuda_requested) {
        if (!contains_provider(impl_->available_providers, "CUDAExecutionProvider")) {
            throw std::runtime_error(
                "CUDAExecutionProvider is not available in this ONNX Runtime build or runtime folder. "
                "Available providers: [" + join_strings(impl_->available_providers) + "]"
            );
        }
        try {
            append_cuda_provider(impl_->session_options, impl_->cuda_config);
            impl_->cuda_registered = true;
        } catch (const Ort::Exception& exc) {
            throw std::runtime_error(
                "Failed to register CUDAExecutionProvider with device_id="
                + std::to_string(impl_->cuda_config.device_id)
                + ", cudnn_conv_algo_search=" + impl_->cuda_config.cudnn_conv_algo_search
                + ": " + std::string(exc.what())
            );
        }
    } else {
        impl_->cpu_fallback_enabled = false;
    }

#ifdef _WIN32
    const std::wstring wide_model_path = to_ort_path(model_path_);
    try {
        impl_->session = std::make_unique<Ort::Session>(impl_->env, wide_model_path.c_str(), impl_->session_options);
    } catch (const Ort::Exception& exc) {
        throw std::runtime_error("Failed to create ONNX Runtime session: " + std::string(exc.what()));
    }
#else
    try {
        impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path_.c_str(), impl_->session_options);
    } catch (const Ort::Exception& exc) {
        throw std::runtime_error("Failed to create ONNX Runtime session: " + std::string(exc.what()));
    }
#endif

    if (impl_->session->GetInputCount() != 1) {
        throw std::runtime_error("Expected exactly one ONNX input.");
    }
    if (impl_->session->GetOutputCount() != 1) {
        throw std::runtime_error("Expected exactly one ONNX output.");
    }

    auto input_name = impl_->session->GetInputNameAllocated(0, impl_->allocator);
    auto output_name = impl_->session->GetOutputNameAllocated(0, impl_->allocator);
    impl_->input_name = input_name.get();
    impl_->output_name = output_name.get();

    impl_->input_shape = impl_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    impl_->output_shape = impl_->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    validate_shape(impl_->input_shape, {1, 3, image_size_, image_size_}, "Input");
    validate_shape(impl_->output_shape, {1, 4 + static_cast<int64_t>(class_names_.size()), 18900}, "Output");
}

InferenceResult OnnxDetector::infer(
    const cv::Mat& image,
    float confidence_threshold,
    float nms_iou_threshold
) {
    if (image.empty()) {
        throw std::invalid_argument("Cannot run inference on an empty image.");
    }

    const auto total_start = std::chrono::steady_clock::now();
    const auto preprocess_start = std::chrono::steady_clock::now();
    PreprocessResult preprocess = preprocess_image(image, image_size_);
    const double preprocess_ms = elapsed_ms(preprocess_start);

    RawInferenceResult raw = run_preprocessed(preprocess);
    const double inference_ms = raw.inference_ms;

    const auto postprocess_start = std::chrono::steady_clock::now();
    std::vector<Detection> detections = decode_yolo_output(
        raw.output.data(),
        raw.output_shape,
        preprocess.letterbox,
        image.size(),
        confidence_threshold,
        nms_iou_threshold,
        class_names_
    );
    const double postprocess_ms = elapsed_ms(postprocess_start);

    InferenceResult result;
    result.is_ng = !detections.empty();
    result.preprocess_ms = preprocess_ms;
    result.inference_ms = inference_ms;
    result.postprocess_ms = postprocess_ms;
    result.total_ms = elapsed_ms(total_start);
    result.provider = impl_->provider;
    result.input_name = impl_->input_name;
    result.output_name = impl_->output_name;
    result.input_shape = impl_->input_shape;
    result.output_shape = raw.output_shape;
    result.detections = std::move(detections);
    return result;
}

RawInferenceResult OnnxDetector::run_preprocessed(PreprocessResult& preprocess) {
    if (preprocess.tensor.empty()) {
        throw std::invalid_argument("Cannot run inference on an empty preprocessed tensor.");
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        preprocess.tensor.data(),
        preprocess.tensor.size(),
        preprocess.shape.data(),
        preprocess.shape.size()
    );

    const char* input_names[] = {impl_->input_name.c_str()};
    const char* output_names[] = {impl_->output_name.c_str()};

    const auto inference_start = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = impl_->session->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );
    const double inference_ms = elapsed_ms(inference_start);

    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error("Expected one tensor output from ONNX Runtime.");
    }
    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> actual_output_shape = output_info.GetShape();
    validate_shape(actual_output_shape, impl_->output_shape, "Runtime output");

    const float* output_data = outputs[0].GetTensorData<float>();
    const std::size_t output_size = static_cast<std::size_t>(output_info.GetElementCount());

    RawInferenceResult result;
    result.inference_ms = inference_ms;
    result.output_shape = actual_output_shape;
    result.output.assign(output_data, output_data + output_size);
    return result;
}

InferenceResult OnnxDetector::infer_preprocessed(
    PreprocessResult& preprocess,
    cv::Size original_size,
    float confidence_threshold,
    float nms_iou_threshold
) {
    const auto total_start = std::chrono::steady_clock::now();
    RawInferenceResult raw = run_preprocessed(preprocess);

    const auto postprocess_start = std::chrono::steady_clock::now();
    std::vector<Detection> detections = decode_yolo_output(
        raw.output.data(),
        raw.output_shape,
        preprocess.letterbox,
        original_size,
        confidence_threshold,
        nms_iou_threshold,
        class_names_
    );
    const double postprocess_ms = elapsed_ms(postprocess_start);

    InferenceResult result;
    result.is_ng = !detections.empty();
    result.preprocess_ms = 0.0;
    result.inference_ms = raw.inference_ms;
    result.postprocess_ms = postprocess_ms;
    result.total_ms = elapsed_ms(total_start);
    result.provider = impl_->provider;
    result.input_name = impl_->input_name;
    result.output_name = impl_->output_name;
    result.input_shape = impl_->input_shape;
    result.output_shape = raw.output_shape;
    result.detections = std::move(detections);
    return result;
}

const std::vector<int64_t>& OnnxDetector::input_shape() const {
    return impl_->input_shape;
}

const std::vector<int64_t>& OnnxDetector::output_shape() const {
    return impl_->output_shape;
}

const std::string& OnnxDetector::input_name() const {
    return impl_->input_name;
}

const std::string& OnnxDetector::output_name() const {
    return impl_->output_name;
}

const std::string& OnnxDetector::provider() const {
    return impl_->provider;
}

const std::vector<std::string>& OnnxDetector::available_providers() const {
    return impl_->available_providers;
}

bool OnnxDetector::cuda_requested() const {
    return impl_->cuda_requested;
}

bool OnnxDetector::cuda_registered() const {
    return impl_->cuda_registered;
}

bool OnnxDetector::cpu_fallback_enabled() const {
    return impl_->cpu_fallback_enabled;
}

const CudaProviderConfig& OnnxDetector::cuda_config() const {
    return impl_->cuda_config;
}

OnnxDetector::~OnnxDetector() = default;

std::vector<std::string> available_execution_providers() {
    return Ort::GetAvailableProviders();
}

}  // namespace pcb_vision
