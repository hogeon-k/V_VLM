#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "detector.hpp"
#include "image_preprocessor.hpp"

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string model;
    std::string metadata;
    std::string image;
    std::string output;
    int imgsz = 960;
    float conf = 0.15F;
    float iou = 0.7F;
    std::string provider = "CUDAExecutionProvider";
    int warmup = 10;
    int repeat = 50;
    std::string cudnn_conv_algo_search = "HEURISTIC";
};

struct TimingStats {
    double first = 0.0;
    double min = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double max = 0.0;
    double standard_deviation = 0.0;
};

struct IterationTiming {
    int iteration = 0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
    bool detections_match = true;
    std::string validation_error;
};

void print_usage(const char* program_name) {
    std::cout
        << "Usage: " << program_name << " --model <best.onnx> --metadata <model_metadata.json> "
        << "--image <image> --output <dir> [--imgsz 960] [--conf 0.15] [--iou 0.7] "
        << "[--provider cuda|cpu] [--warmup 10] [--repeat 50] "
        << "[--cudnn-conv-algo-search heuristic|exhaustive|default]\n"
        << "  --cudnn-conv-algo-search: allowed values: heuristic, exhaustive, default; "
        << "default: heuristic\n";
}

std::string require_value(int& index, int argc, char* argv[], const std::string& option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(option + " requires a value.");
    }
    return argv[++index];
}

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (option == "--model") {
            args.model = require_value(index, argc, argv, option);
        } else if (option == "--metadata") {
            args.metadata = require_value(index, argc, argv, option);
        } else if (option == "--image") {
            args.image = require_value(index, argc, argv, option);
        } else if (option == "--output") {
            args.output = require_value(index, argc, argv, option);
        } else if (option == "--imgsz") {
            args.imgsz = std::stoi(require_value(index, argc, argv, option));
        } else if (option == "--conf") {
            args.conf = std::stof(require_value(index, argc, argv, option));
        } else if (option == "--iou") {
            args.iou = std::stof(require_value(index, argc, argv, option));
        } else if (option == "--provider") {
            std::string provider = require_value(index, argc, argv, option);
            std::transform(provider.begin(), provider.end(), provider.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });
            if (provider == "cuda" || provider == "cudaexecutionprovider") {
                args.provider = "CUDAExecutionProvider";
            } else if (provider == "cpu" || provider == "cpuexecutionprovider") {
                args.provider = "CPUExecutionProvider";
            } else {
                throw std::invalid_argument("--provider must be cuda or cpu.");
            }
        } else if (option == "--warmup") {
            args.warmup = std::stoi(require_value(index, argc, argv, option));
        } else if (option == "--repeat") {
            args.repeat = std::stoi(require_value(index, argc, argv, option));
        } else if (option == "--cudnn-conv-algo-search") {
            const std::string value = index + 1 < argc ? argv[++index] : "";
            args.cudnn_conv_algo_search = pcb_vision::normalize_cudnn_conv_algo_search(
                value
            );
        } else {
            throw std::invalid_argument("Unknown argument: " + option);
        }
    }
    if (args.model.empty() || args.image.empty() || args.output.empty()) {
        throw std::invalid_argument("--model, --image, and --output are required.");
    }
    if (args.metadata.empty()) {
        args.metadata = "models/model_metadata.json";
    }
    if (args.warmup < 0) {
        throw std::invalid_argument("--warmup must be 0 or greater.");
    }
    if (args.repeat <= 0) {
        throw std::invalid_argument("--repeat must be greater than 0.");
    }
    return args;
}

std::string read_text(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

std::vector<std::string> fallback_class_names() {
    return {"open_circuit", "short", "missing_hole"};
}

std::vector<std::string> load_class_names(const std::string& metadata_path) {
    if (metadata_path.empty() || !fs::exists(metadata_path)) {
        return fallback_class_names();
    }
    const std::string text = read_text(metadata_path);
    const std::string key = "\"class_names\"";
    const std::size_t key_pos = text.find(key);
    if (key_pos == std::string::npos) {
        return fallback_class_names();
    }
    const std::size_t open = text.find('[', key_pos);
    const std::size_t close = text.find(']', open);
    if (open == std::string::npos || close == std::string::npos || close <= open) {
        return fallback_class_names();
    }
    const std::string array_text = text.substr(open + 1, close - open - 1);
    std::vector<std::string> names;
    std::size_t pos = 0;
    while (true) {
        const std::size_t quote_start = array_text.find('"', pos);
        if (quote_start == std::string::npos) {
            break;
        }
        const std::size_t quote_end = array_text.find('"', quote_start + 1);
        if (quote_end == std::string::npos) {
            break;
        }
        names.push_back(array_text.substr(quote_start + 1, quote_end - quote_start - 1));
        pos = quote_end + 1;
    }
    return names.empty() ? fallback_class_names() : names;
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (char ch : value) {
        switch (ch) {
            case '\\': escaped << "\\\\"; break;
            case '"': escaped << "\\\""; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default: escaped << ch; break;
        }
    }
    return escaped.str();
}

std::string shape_to_json(const std::vector<int64_t>& shape) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
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

TimingStats calculate_stats(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot calculate timing statistics for an empty sample.");
    }

    TimingStats stats;
    stats.first = values.front();
    stats.min = *std::min_element(values.begin(), values.end());
    stats.max = *std::max_element(values.begin(), values.end());
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t middle = sorted.size() / 2;
    if (sorted.size() % 2 == 0) {
        stats.median = (sorted[middle - 1] + sorted[middle]) / 2.0;
    } else {
        stats.median = sorted[middle];
    }
    const std::size_t p95_index = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(sorted.size()))
    ) - 1;
    stats.p95 = sorted[std::min(p95_index, sorted.size() - 1)];

    double squared_sum = 0.0;
    for (double value : values) {
        const double diff = value - stats.mean;
        squared_sum += diff * diff;
    }
    stats.standard_deviation = std::sqrt(squared_sum / static_cast<double>(values.size()));
    return stats;
}

std::string validate_detections(
    const std::vector<pcb_vision::Detection>& expected,
    const std::vector<pcb_vision::Detection>& actual
) {
    if (expected.size() != actual.size()) {
        return "detection count mismatch: expected " + std::to_string(expected.size())
            + ", actual " + std::to_string(actual.size());
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const auto& left = expected[index];
        const auto& right = actual[index];
        if (left.class_id != right.class_id) {
            return "class_id mismatch at detection " + std::to_string(index);
        }
        if (std::fabs(left.confidence - right.confidence) > 0.001F) {
            return "confidence mismatch at detection " + std::to_string(index);
        }
        const float left_x2 = left.box.x + left.box.width;
        const float left_y2 = left.box.y + left.box.height;
        const float right_x2 = right.box.x + right.box.width;
        const float right_y2 = right.box.y + right.box.height;
        if (std::fabs(left.box.x - right.box.x) > 1.0F
            || std::fabs(left.box.y - right.box.y) > 1.0F
            || std::fabs(left_x2 - right_x2) > 1.0F
            || std::fabs(left_y2 - right_y2) > 1.0F) {
            return "bbox mismatch at detection " + std::to_string(index);
        }
    }
    return "";
}

void write_stats_json(std::ostream& out, const std::string& name, const TimingStats& stats, const std::string& indent) {
    out << indent << "\"" << name << "\": {"
        << "\"first\": " << stats.first
        << ", \"min\": " << stats.min
        << ", \"mean\": " << stats.mean
        << ", \"median\": " << stats.median
        << ", \"p95\": " << stats.p95
        << ", \"max\": " << stats.max
        << ", \"standard_deviation\": " << stats.standard_deviation
        << "}";
}

void write_json(const fs::path& path, const Args& args, const pcb_vision::InferenceResult& result) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write JSON: " + path.string());
    }
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"model\": \"" << json_escape(args.model) << "\",\n";
    out << "  \"image\": \"" << json_escape(args.image) << "\",\n";
    out << "  \"provider\": \"" << json_escape(result.provider) << "\",\n";
    out << "  \"input_name\": \"" << json_escape(result.input_name) << "\",\n";
    out << "  \"output_name\": \"" << json_escape(result.output_name) << "\",\n";
    out << "  \"input_shape\": " << shape_to_json(result.input_shape) << ",\n";
    out << "  \"output_shape\": " << shape_to_json(result.output_shape) << ",\n";
    out << "  \"config\": {\"imgsz\": " << args.imgsz << ", \"conf\": " << args.conf
        << ", \"iou\": " << args.iou << ", \"warmup\": " << args.warmup
        << ", \"repeat\": " << args.repeat
        << ", \"cudnn_conv_algo_search\": \"" << json_escape(args.cudnn_conv_algo_search) << "\"},\n";
    out << "  \"timing_ms\": {\"preprocess\": " << result.preprocess_ms
        << ", \"inference\": " << result.inference_ms
        << ", \"postprocess\": " << result.postprocess_ms
        << ", \"total\": " << result.total_ms << "},\n";
    out << "  \"detections\": [\n";
    for (std::size_t i = 0; i < result.detections.size(); ++i) {
        const auto& detection = result.detections[i];
        out << "    {\"class_id\": " << detection.class_id
            << ", \"class_name\": \"" << json_escape(detection.class_name) << "\""
            << ", \"confidence\": " << detection.confidence
            << ", \"bbox\": [" << detection.box.x << ", " << detection.box.y << ", "
            << detection.box.x + detection.box.width << ", " << detection.box.y + detection.box.height << "]}";
        out << (i + 1 < result.detections.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

void write_benchmark_json(
    const fs::path& path,
    const Args& args,
    const std::vector<std::string>& available_providers,
    const pcb_vision::OnnxDetector& detector,
    const TimingStats& run_stats,
    const TimingStats& preprocess_stats,
    const TimingStats& e2e_inference_stats,
    const TimingStats& postprocess_stats,
    const TimingStats& total_stats,
    const std::vector<IterationTiming>& run_iterations,
    const std::vector<IterationTiming>& e2e_iterations
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write benchmark JSON: " + path.string());
    }
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"model\": \"" << json_escape(args.model) << "\",\n";
    out << "  \"image\": \"" << json_escape(args.image) << "\",\n";
    out << "  \"provider\": \"" << json_escape(args.provider) << "\",\n";
    out << "  \"available_providers\": [";
    for (std::size_t i = 0; i < available_providers.size(); ++i) {
        out << (i > 0 ? ", " : "") << "\"" << json_escape(available_providers[i]) << "\"";
    }
    out << "],\n";
    out << "  \"cuda\": {\"registered\": " << (detector.cuda_registered() ? "true" : "false")
        << ", \"device_id\": " << detector.cuda_config().device_id
        << ", \"cudnn_conv_algo_search\": \"" << json_escape(detector.cuda_config().cudnn_conv_algo_search) << "\"},\n";
    out << "  \"config\": {\"imgsz\": " << args.imgsz << ", \"conf\": " << args.conf
        << ", \"iou\": " << args.iou << ", \"warmup\": " << args.warmup
        << ", \"repeat\": " << args.repeat << "},\n";
    out << "  \"session_run_ms\": {\n";
    write_stats_json(out, "stats", run_stats, "    ");
    out << "\n  },\n";
    out << "  \"end_to_end_ms\": {\n";
    write_stats_json(out, "preprocess", preprocess_stats, "    ");
    out << ",\n";
    write_stats_json(out, "inference", e2e_inference_stats, "    ");
    out << ",\n";
    write_stats_json(out, "postprocess", postprocess_stats, "    ");
    out << ",\n";
    write_stats_json(out, "total", total_stats, "    ");
    out << "\n  },\n";
    out << "  \"validation\": {\"run_only_mismatches\": ";
    out << std::count_if(run_iterations.begin(), run_iterations.end(), [](const IterationTiming& item) {
        return !item.detections_match;
    });
    out << ", \"end_to_end_mismatches\": ";
    out << std::count_if(e2e_iterations.begin(), e2e_iterations.end(), [](const IterationTiming& item) {
        return !item.detections_match;
    });
    out << "}\n";
    out << "}\n";
}

void write_benchmark_csv(
    const fs::path& path,
    const std::vector<IterationTiming>& run_iterations,
    const std::vector<IterationTiming>& e2e_iterations
) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to write benchmark CSV: " + path.string());
    }
    out << "phase,iteration,preprocess_ms,inference_ms,postprocess_ms,total_ms,detections_match,validation_error\n";
    const auto write_rows = [&out](const std::string& phase, const std::vector<IterationTiming>& rows) {
        for (const IterationTiming& row : rows) {
            out << phase << ','
                << row.iteration << ','
                << row.preprocess_ms << ','
                << row.inference_ms << ','
                << row.postprocess_ms << ','
                << row.total_ms << ','
                << (row.detections_match ? "true" : "false") << ','
                << '"' << row.validation_error << '"' << '\n';
        }
    };
    out << std::fixed << std::setprecision(6);
    write_rows("session_run", run_iterations);
    write_rows("end_to_end", e2e_iterations);
}

void write_csv(const fs::path& path, const std::vector<pcb_vision::Detection>& detections) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to write CSV: " + path.string());
    }
    const unsigned char bom[] = {0xEF, 0xBB, 0xBF};
    out.write(reinterpret_cast<const char*>(bom), 3);
    out << "index,class_id,class_name,confidence,x1,y1,x2,y2\n";
    out << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < detections.size(); ++i) {
        const auto& detection = detections[i];
        out << i << ','
            << detection.class_id << ','
            << detection.class_name << ','
            << detection.confidence << ','
            << detection.box.x << ','
            << detection.box.y << ','
            << detection.box.x + detection.box.width << ','
            << detection.box.y + detection.box.height << '\n';
    }
}

void draw_result_image(const fs::path& path, cv::Mat image, const std::vector<pcb_vision::Detection>& detections) {
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(40, 180, 255),
        cv::Scalar(80, 220, 90),
        cv::Scalar(230, 90, 120)
    };
    for (const auto& detection : detections) {
        const cv::Scalar color = colors[static_cast<std::size_t>(std::max(0, detection.class_id)) % colors.size()];
        const cv::Rect rect(
            static_cast<int>(std::round(detection.box.x)),
            static_cast<int>(std::round(detection.box.y)),
            static_cast<int>(std::round(detection.box.width)),
            static_cast<int>(std::round(detection.box.height))
        );
        cv::rectangle(image, rect, color, 2);
        std::ostringstream label;
        label << detection.class_name << " " << std::fixed << std::setprecision(3) << detection.confidence;
        cv::putText(image, label.str(), cv::Point(rect.x, std::max(16, rect.y - 6)), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
    }
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Failed to write result image: " + path.string());
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const Args args = parse_args(argc, argv);
        const std::vector<std::string> available_providers = pcb_vision::available_execution_providers();
        const std::vector<std::string> class_names = load_class_names(args.metadata);
        const cv::Mat image = pcb_vision::load_bgr_image(args.image);
        fs::create_directories(args.output);

        pcb_vision::CudaProviderConfig cuda_config;
        cuda_config.device_id = 0;
        cuda_config.cudnn_conv_algo_search = args.cudnn_conv_algo_search;
        pcb_vision::OnnxDetector detector(args.model, class_names, args.imgsz, args.provider, cuda_config);

        pcb_vision::PreprocessResult benchmark_input = pcb_vision::preprocess_image(image, args.imgsz);
        for (int iteration = 0; iteration < args.warmup; ++iteration) {
            (void)detector.infer_preprocessed(benchmark_input, image.size(), args.conf, args.iou);
        }

        std::vector<IterationTiming> run_iterations;
        run_iterations.reserve(static_cast<std::size_t>(args.repeat));
        std::vector<double> run_times;
        run_times.reserve(static_cast<std::size_t>(args.repeat));
        pcb_vision::InferenceResult result;
        bool has_baseline = false;
        for (int iteration = 0; iteration < args.repeat; ++iteration) {
            pcb_vision::InferenceResult current = detector.infer_preprocessed(benchmark_input, image.size(), args.conf, args.iou);
            if (!has_baseline) {
                result = current;
                has_baseline = true;
            }
            const std::string validation_error = validate_detections(result.detections, current.detections);
            run_iterations.push_back(IterationTiming{
                iteration,
                current.preprocess_ms,
                current.inference_ms,
                current.postprocess_ms,
                current.total_ms,
                validation_error.empty(),
                validation_error
            });
            run_times.push_back(current.inference_ms);
        }

        std::vector<IterationTiming> e2e_iterations;
        e2e_iterations.reserve(static_cast<std::size_t>(args.repeat));
        std::vector<double> preprocess_times;
        std::vector<double> e2e_inference_times;
        std::vector<double> postprocess_times;
        std::vector<double> total_times;
        preprocess_times.reserve(static_cast<std::size_t>(args.repeat));
        e2e_inference_times.reserve(static_cast<std::size_t>(args.repeat));
        postprocess_times.reserve(static_cast<std::size_t>(args.repeat));
        total_times.reserve(static_cast<std::size_t>(args.repeat));
        for (int iteration = 0; iteration < args.repeat; ++iteration) {
            pcb_vision::InferenceResult current = detector.infer(image, args.conf, args.iou);
            const std::string validation_error = validate_detections(result.detections, current.detections);
            e2e_iterations.push_back(IterationTiming{
                iteration,
                current.preprocess_ms,
                current.inference_ms,
                current.postprocess_ms,
                current.total_ms,
                validation_error.empty(),
                validation_error
            });
            preprocess_times.push_back(current.preprocess_ms);
            e2e_inference_times.push_back(current.inference_ms);
            postprocess_times.push_back(current.postprocess_ms);
            total_times.push_back(current.total_ms);
        }

        const fs::path output_dir(args.output);
        write_json(output_dir / "result.json", args, result);
        write_csv(output_dir / "detections.csv", result.detections);
        draw_result_image(output_dir / "result.jpg", image.clone(), result.detections);
        const TimingStats run_stats = calculate_stats(run_times);
        const TimingStats preprocess_stats = calculate_stats(preprocess_times);
        const TimingStats e2e_inference_stats = calculate_stats(e2e_inference_times);
        const TimingStats postprocess_stats = calculate_stats(postprocess_times);
        const TimingStats total_stats = calculate_stats(total_times);
        write_benchmark_json(
            output_dir / "benchmark.json",
            args,
            available_providers,
            detector,
            run_stats,
            preprocess_stats,
            e2e_inference_stats,
            postprocess_stats,
            total_stats,
            run_iterations,
            e2e_iterations
        );
        write_benchmark_csv(output_dir / "benchmark.csv", run_iterations, e2e_iterations);

        std::cout << "=== C++ ONNX Runtime Inference ===\n";
        std::cout << "Model: " << args.model << '\n';
        std::cout << "Image: " << args.image << '\n';
        std::cout << "Available providers: [" << join_strings(available_providers) << "]\n";
        std::cout << "Requested provider: " << args.provider << '\n';
        if (detector.cuda_requested()) {
            std::cout << "CUDA registration: "
                      << (detector.cuda_registered() ? "success" : "failed")
                      << '\n';
            std::cout << "CUDA device id: " << detector.cuda_config().device_id << '\n';
            std::cout << "cuDNN convolution algorithm search: "
                      << detector.cuda_config().cudnn_conv_algo_search << '\n';
        }
        std::cout << "CPU fallback: "
                  << (detector.cpu_fallback_enabled() ? "enabled by ONNX Runtime after CUDA provider" : "disabled/not used in CPU-only mode")
                  << '\n';
        std::cout << "Warmup: " << args.warmup << " runs excluded from statistics\n";
        std::cout << "Repeat: " << args.repeat << " measured runs\n";
        std::cout << "Input shape: " << shape_to_json(result.input_shape) << '\n';
        std::cout << "Output shape: " << shape_to_json(result.output_shape) << '\n';
        std::cout << "Provider: " << result.provider << "\n\n";
        std::cout << "Detections: " << result.detections.size() << "\n\n";
        for (std::size_t i = 0; i < result.detections.size(); ++i) {
            const auto& detection = result.detections[i];
            std::cout << "[" << i + 1 << "]\n";
            std::cout << "class_id: " << detection.class_id << '\n';
            std::cout << "class_name: " << detection.class_name << '\n';
            std::cout << "confidence: " << detection.confidence << '\n';
            std::cout << "bbox: [" << detection.box.x << ", " << detection.box.y << ", "
                      << detection.box.x + detection.box.width << ", " << detection.box.y + detection.box.height << "]\n\n";
        }
        std::cout << "Session.Run stats (ms): first=" << run_stats.first
                  << ", min=" << run_stats.min
                  << ", mean=" << run_stats.mean
                  << ", median=" << run_stats.median
                  << ", p95=" << run_stats.p95
                  << ", max=" << run_stats.max
                  << ", stddev=" << run_stats.standard_deviation << '\n';
        std::cout << "End-to-end total stats (ms): first=" << total_stats.first
                  << ", min=" << total_stats.min
                  << ", mean=" << total_stats.mean
                  << ", median=" << total_stats.median
                  << ", p95=" << total_stats.p95
                  << ", max=" << total_stats.max
                  << ", stddev=" << total_stats.standard_deviation << '\n';
        std::cout << "End-to-end mean breakdown (ms): preprocess=" << preprocess_stats.mean
                  << ", inference=" << e2e_inference_stats.mean
                  << ", postprocess=" << postprocess_stats.mean
                  << ", total=" << total_stats.mean << '\n';
        std::cout << "Validation mismatches: session_run="
                  << std::count_if(run_iterations.begin(), run_iterations.end(), [](const IterationTiming& item) {
                         return !item.detections_match;
                     })
                  << ", end_to_end="
                  << std::count_if(e2e_iterations.begin(), e2e_iterations.end(), [](const IterationTiming& item) {
                         return !item.detections_match;
                     })
                  << '\n';
        std::cout << "Output: " << output_dir.string() << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << '\n';
        return 1;
    }
}
