#include "detector.hpp"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <stdexcept>
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
};

OnnxDetector::OnnxDetector(
    std::string model_path,
    std::vector<std::string> class_names,
    int image_size
) : impl_(std::make_unique<Impl>()),
    model_path_(std::move(model_path)),
    class_names_(std::move(class_names)),
    image_size_(image_size) {
    if (class_names_.empty()) {
        throw std::invalid_argument("class_names must not be empty.");
    }
    impl_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
    const std::wstring wide_model_path = to_ort_path(model_path_);
    impl_->session = std::make_unique<Ort::Session>(impl_->env, wide_model_path.c_str(), impl_->session_options);
#else
    impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path_.c_str(), impl_->session_options);
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

    const auto postprocess_start = std::chrono::steady_clock::now();
    const float* output_data = outputs[0].GetTensorData<float>();
    std::vector<Detection> detections = decode_yolo_output(
        output_data,
        actual_output_shape,
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
    result.output_shape = actual_output_shape;
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

OnnxDetector::~OnnxDetector() = default;

}  // namespace pcb_vision
