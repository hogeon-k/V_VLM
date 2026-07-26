#include "tensorrt_detector.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "postprocessor.hpp"
#include "unicode_utils.hpp"

namespace pcb_vision {
namespace {

double elapsed_ms(const std::chrono::steady_clock::time_point& start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << severity_label(severity) << ": " << message << '\n';
        }
    }

private:
    static const char* severity_label(Severity severity) noexcept {
        switch (severity) {
            case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
            case Severity::kERROR: return "ERROR";
            case Severity::kWARNING: return "WARNING";
            case Severity::kINFO: return "INFO";
            case Severity::kVERBOSE: return "VERBOSE";
            default: return "UNKNOWN";
        }
    }
};

void check_cuda(cudaError_t status, const std::string& action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(action + " failed: " + cudaGetErrorString(status));
    }
}

struct CudaStreamDeleter {
    void operator()(cudaStream_t stream) const noexcept {
        if (stream != nullptr) {
            (void)cudaStreamDestroy(stream);
        }
    }
};

struct CudaEventDeleter {
    void operator()(cudaEvent_t event) const noexcept {
        if (event != nullptr) {
            (void)cudaEventDestroy(event);
        }
    }
};

struct CudaBuffer {
    void* data = nullptr;
    std::size_t bytes = 0;

    CudaBuffer() = default;
    explicit CudaBuffer(std::size_t requested_bytes) : bytes(requested_bytes) {
        check_cuda(cudaMalloc(&data, bytes), "cudaMalloc(" + std::to_string(bytes) + " bytes)");
    }
    ~CudaBuffer() {
        if (data != nullptr) {
            (void)cudaFree(data);
        }
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept : data(other.data), bytes(other.bytes) {
        other.data = nullptr;
        other.bytes = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (data != nullptr) {
                (void)cudaFree(data);
            }
            data = other.data;
            bytes = other.bytes;
            other.data = nullptr;
            other.bytes = 0;
        }
        return *this;
    }
};

using StreamPtr = std::unique_ptr<std::remove_pointer<cudaStream_t>::type, CudaStreamDeleter>;
using EventPtr = std::unique_ptr<std::remove_pointer<cudaEvent_t>::type, CudaEventDeleter>;

StreamPtr make_stream() {
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    return StreamPtr(stream);
}

EventPtr make_event() {
    cudaEvent_t event = nullptr;
    check_cuda(cudaEventCreate(&event), "cudaEventCreate");
    return EventPtr(event);
}

std::vector<unsigned char> read_engine_file(const std::string& path) {
    std::vector<unsigned char> data = read_binary_file(path_from_utf8(path));
    if (data.empty()) {
        throw std::runtime_error("TensorRT engine file is empty: " + path);
    }
    return data;
}

std::vector<int64_t> dims_to_shape(const nvinfer1::Dims& dims) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int index = 0; index < dims.nbDims; ++index) {
        shape.push_back(static_cast<int64_t>(dims.d[index]));
    }
    return shape;
}

std::size_t element_count(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("Tensor shape must not be empty.");
    }
    std::size_t count = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("Dynamic or invalid TensorRT tensor shape is not supported.");
        }
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

std::string dtype_to_string(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return "FP32";
        case nvinfer1::DataType::kHALF: return "FP16";
        case nvinfer1::DataType::kINT8: return "INT8";
        case nvinfer1::DataType::kINT32: return "INT32";
        case nvinfer1::DataType::kBOOL: return "BOOL";
        default: return "UNKNOWN";
    }
}

std::size_t dtype_byte_size(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
        default: throw std::runtime_error("Unsupported TensorRT tensor dtype: " + dtype_to_string(dtype));
    }
}

void validate_shape(
    const std::vector<int64_t>& actual,
    const std::vector<int64_t>& expected,
    const std::string& label
) {
    if (actual != expected) {
        std::ostringstream message;
        message << label << " shape mismatch. Expected [";
        for (std::size_t i = 0; i < expected.size(); ++i) {
            message << (i > 0 ? ", " : "") << expected[i];
        }
        message << "], actual [";
        for (std::size_t i = 0; i < actual.size(); ++i) {
            message << (i > 0 ? ", " : "") << actual[i];
        }
        message << "].";
        throw std::runtime_error(message.str());
    }
}

}  // namespace

struct TensorRtDetector::Impl {
    TensorRtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    StreamPtr stream;
    EventPtr h2d_start;
    EventPtr h2d_end;
    EventPtr gpu_start;
    EventPtr gpu_end;
    EventPtr d2h_start;
    EventPtr d2h_end;
    CudaBuffer input_buffer;
    CudaBuffer output_buffer;
    std::vector<float> host_output;
    TensorRtTensorInfo input_info;
    TensorRtTensorInfo output_info;
    TensorRtRunTiming last_timing;
    std::string engine_path;
    std::vector<std::string> class_names;
    int image_size = 960;
    int device_id = 0;
};

TensorRtDetector::TensorRtDetector(
    std::string engine_path,
    std::vector<std::string> class_names,
    int image_size,
    int device_id
) : impl_(std::make_unique<Impl>()) {
    if (class_names.empty()) {
        throw std::invalid_argument("class_names must not be empty.");
    }
    impl_->engine_path = std::move(engine_path);
    impl_->class_names = std::move(class_names);
    impl_->image_size = image_size;
    impl_->device_id = device_id;

    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice(" + std::to_string(impl_->device_id) + ")");

    const std::vector<unsigned char> engine_bytes = read_engine_file(impl_->engine_path);
    impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
    if (!impl_->runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }
    impl_->engine.reset(impl_->runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
    if (!impl_->engine) {
        throw std::runtime_error("Failed to deserialize TensorRT engine: " + impl_->engine_path);
    }
    impl_->context.reset(impl_->engine->createExecutionContext());
    if (!impl_->context) {
        throw std::runtime_error("Failed to create TensorRT execution context.");
    }

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    const int io_count = impl_->engine->getNbIOTensors();
    for (int index = 0; index < io_count; ++index) {
        const char* tensor_name = impl_->engine->getIOTensorName(index);
        if (tensor_name == nullptr) {
            throw std::runtime_error("TensorRT engine returned a null tensor name.");
        }
        const std::string name(tensor_name);
        const nvinfer1::TensorIOMode mode = impl_->engine->getTensorIOMode(tensor_name);
        const nvinfer1::DataType dtype = impl_->engine->getTensorDataType(tensor_name);
        TensorRtTensorInfo info{name, dims_to_shape(impl_->engine->getTensorShape(tensor_name)), dtype_to_string(dtype)};
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_names.push_back(name);
            impl_->input_info = std::move(info);
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_names.push_back(name);
            impl_->output_info = std::move(info);
        }
    }

    if (input_names.size() != 1) {
        throw std::runtime_error("Expected exactly one TensorRT input tensor.");
    }
    if (output_names.size() != 1) {
        throw std::runtime_error("Expected exactly one TensorRT output tensor.");
    }
    if (impl_->input_info.name != "images") {
        throw std::runtime_error("TensorRT input tensor name mismatch. Expected images, actual " + impl_->input_info.name + ".");
    }
    if (impl_->output_info.name != "output0") {
        throw std::runtime_error("TensorRT output tensor name mismatch. Expected output0, actual " + impl_->output_info.name + ".");
    }
    validate_shape(impl_->input_info.shape, {1, 3, image_size, image_size}, "TensorRT input");
    validate_shape(
        impl_->output_info.shape,
        {1, 4 + static_cast<int64_t>(impl_->class_names.size()), 18900},
        "TensorRT output"
    );
    if (impl_->input_info.dtype != "FP32") {
        throw std::runtime_error("Unsupported TensorRT input dtype: " + impl_->input_info.dtype + ". Expected FP32.");
    }
    if (impl_->output_info.dtype != "FP32") {
        throw std::runtime_error("Unsupported TensorRT output dtype: " + impl_->output_info.dtype + ". Expected FP32.");
    }

    const std::size_t input_bytes = element_count(impl_->input_info.shape) * dtype_byte_size(nvinfer1::DataType::kFLOAT);
    const std::size_t output_count = element_count(impl_->output_info.shape);
    const std::size_t output_bytes = output_count * dtype_byte_size(nvinfer1::DataType::kFLOAT);
    impl_->input_buffer = CudaBuffer(input_bytes);
    impl_->output_buffer = CudaBuffer(output_bytes);
    impl_->host_output.resize(output_count);
    impl_->stream = make_stream();
    impl_->h2d_start = make_event();
    impl_->h2d_end = make_event();
    impl_->gpu_start = make_event();
    impl_->gpu_end = make_event();
    impl_->d2h_start = make_event();
    impl_->d2h_end = make_event();
}

TensorRtDetector::~TensorRtDetector() = default;
TensorRtDetector::TensorRtDetector(TensorRtDetector&&) noexcept = default;
TensorRtDetector& TensorRtDetector::operator=(TensorRtDetector&&) noexcept = default;

InferenceResult TensorRtDetector::infer(
    const cv::Mat& image,
    float confidence_threshold,
    float nms_iou_threshold
) {
    if (image.empty()) {
        throw std::invalid_argument("Cannot run inference on an empty image.");
    }

    const auto total_start = std::chrono::steady_clock::now();
    const auto preprocess_start = std::chrono::steady_clock::now();
    PreprocessResult preprocess = preprocess_image(image, impl_->image_size);
    const double preprocess_ms = elapsed_ms(preprocess_start);

    RawInferenceResult raw = run_preprocessed(preprocess);

    const auto postprocess_start = std::chrono::steady_clock::now();
    std::vector<Detection> detections = decode_yolo_output(
        raw.output.data(),
        raw.output_shape,
        preprocess.letterbox,
        image.size(),
        confidence_threshold,
        nms_iou_threshold,
        impl_->class_names
    );
    const double postprocess_ms = elapsed_ms(postprocess_start);

    InferenceResult result;
    result.is_ng = !detections.empty();
    result.preprocess_ms = preprocess_ms;
    result.inference_ms = raw.inference_ms;
    result.postprocess_ms = postprocess_ms;
    result.total_ms = elapsed_ms(total_start);
    result.provider = "Native TensorRT";
    result.input_name = impl_->input_info.name;
    result.output_name = impl_->output_info.name;
    result.input_shape = impl_->input_info.shape;
    result.output_shape = raw.output_shape;
    result.detections = std::move(detections);
    return result;
}

RawInferenceResult TensorRtDetector::run_preprocessed(PreprocessResult& preprocess) {
    if (preprocess.tensor.empty()) {
        throw std::invalid_argument("Cannot run inference on an empty preprocessed tensor.");
    }
    const std::size_t input_bytes = preprocess.tensor.size() * sizeof(float);
    if (input_bytes != impl_->input_buffer.bytes) {
        throw std::runtime_error("Preprocessed tensor size does not match TensorRT input buffer size.");
    }

    const auto total_start = std::chrono::steady_clock::now();
    check_cuda(cudaEventRecord(impl_->h2d_start.get(), impl_->stream.get()), "cudaEventRecord h2d_start");
    check_cuda(
        cudaMemcpyAsync(impl_->input_buffer.data, preprocess.tensor.data(), input_bytes, cudaMemcpyHostToDevice, impl_->stream.get()),
        "cudaMemcpyAsync Host to Device"
    );
    check_cuda(cudaEventRecord(impl_->h2d_end.get(), impl_->stream.get()), "cudaEventRecord h2d_end");
    check_cuda(cudaEventSynchronize(impl_->h2d_end.get()), "cudaEventSynchronize h2d_end");
    float h2d_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&h2d_ms, impl_->h2d_start.get(), impl_->h2d_end.get()), "cudaEventElapsedTime H2D");

    if (!impl_->context->setTensorAddress(impl_->input_info.name.c_str(), impl_->input_buffer.data)) {
        throw std::runtime_error("Failed to set TensorRT input tensor address: " + impl_->input_info.name);
    }
    if (!impl_->context->setTensorAddress(impl_->output_info.name.c_str(), impl_->output_buffer.data)) {
        throw std::runtime_error("Failed to set TensorRT output tensor address: " + impl_->output_info.name);
    }

    check_cuda(cudaEventRecord(impl_->gpu_start.get(), impl_->stream.get()), "cudaEventRecord gpu_start");
    if (!impl_->context->enqueueV3(impl_->stream.get())) {
        throw std::runtime_error("TensorRT enqueueV3 failed.");
    }
    check_cuda(cudaEventRecord(impl_->gpu_end.get(), impl_->stream.get()), "cudaEventRecord gpu_end");
    check_cuda(cudaEventSynchronize(impl_->gpu_end.get()), "cudaEventSynchronize gpu_end");
    float gpu_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&gpu_ms, impl_->gpu_start.get(), impl_->gpu_end.get()), "cudaEventElapsedTime");

    check_cuda(cudaEventRecord(impl_->d2h_start.get(), impl_->stream.get()), "cudaEventRecord d2h_start");
    check_cuda(
        cudaMemcpyAsync(
            impl_->host_output.data(),
            impl_->output_buffer.data,
            impl_->host_output.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            impl_->stream.get()
        ),
        "cudaMemcpyAsync Device to Host"
    );
    check_cuda(cudaEventRecord(impl_->d2h_end.get(), impl_->stream.get()), "cudaEventRecord d2h_end");
    check_cuda(cudaEventSynchronize(impl_->d2h_end.get()), "cudaEventSynchronize d2h_end");
    float d2h_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&d2h_ms, impl_->d2h_start.get(), impl_->d2h_end.get()), "cudaEventElapsedTime D2H");

    RawInferenceResult result;
    result.inference_ms = static_cast<double>(gpu_ms);
    result.output_shape = impl_->output_info.shape;
    result.output = impl_->host_output;

    impl_->last_timing.h2d_ms = static_cast<double>(h2d_ms);
    impl_->last_timing.gpu_execution_ms = static_cast<double>(gpu_ms);
    impl_->last_timing.d2h_ms = static_cast<double>(d2h_ms);
    impl_->last_timing.total_ms = elapsed_ms(total_start);
    return result;
}

InferenceResult TensorRtDetector::infer_preprocessed(
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
        impl_->class_names
    );
    const double postprocess_ms = elapsed_ms(postprocess_start);

    InferenceResult result;
    result.is_ng = !detections.empty();
    result.preprocess_ms = 0.0;
    result.inference_ms = raw.inference_ms;
    result.postprocess_ms = postprocess_ms;
    result.total_ms = elapsed_ms(total_start);
    result.provider = "Native TensorRT";
    result.input_name = impl_->input_info.name;
    result.output_name = impl_->output_info.name;
    result.input_shape = impl_->input_info.shape;
    result.output_shape = raw.output_shape;
    result.detections = std::move(detections);
    return result;
}

const TensorRtTensorInfo& TensorRtDetector::input_info() const {
    return impl_->input_info;
}

const TensorRtTensorInfo& TensorRtDetector::output_info() const {
    return impl_->output_info;
}

const TensorRtRunTiming& TensorRtDetector::last_timing() const {
    return impl_->last_timing;
}

const std::string& TensorRtDetector::engine_path() const {
    return impl_->engine_path;
}

int TensorRtDetector::device_id() const {
    return impl_->device_id;
}

std::string TensorRtDetector::version_string() const {
    std::ostringstream out;
    out << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH;
    return out.str();
}

}  // namespace pcb_vision
