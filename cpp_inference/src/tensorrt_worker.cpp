#include "tensorrt_worker.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "image_preprocessor.hpp"
#include "tensorrt_detector.hpp"
#include "unicode_utils.hpp"

namespace fs = std::filesystem;

namespace pcb_vision {
namespace {

struct WorkerRequest {
    std::string request_id;
    std::string command;
    std::string image;
    std::string output;
    float confidence = 0.0F;
    float iou = 0.0F;
};

double elapsed_ms(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start
    ).count();
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': escaped << "\\\\"; break;
            case '"': escaped << "\\\""; break;
            case '\b': escaped << "\\b"; break;
            case '\f': escaped << "\\f"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (ch < 0x20) {
                    escaped << "\\u"
                            << std::hex << std::setw(4) << std::setfill('0')
                            << static_cast<int>(ch)
                            << std::dec << std::setfill(' ');
                } else {
                    escaped << static_cast<char>(ch);
                }
                break;
        }
    }
    return escaped.str();
}

void write_protocol_line(const std::string& json) {
    std::cout << json << '\n' << std::flush;
}

std::string request_id_from_line(const std::string& line) {
    try {
        cv::FileStorage storage(line, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
        const cv::FileNode node = storage.root()["request_id"];
        if (!node.empty() && node.isString()) {
            return static_cast<std::string>(node);
        }
    } catch (const cv::Exception&) {
    }
    return "";
}

std::string required_string(const cv::FileNode& root, const char* name) {
    const cv::FileNode node = root[name];
    if (node.empty() || !node.isString()) {
        throw std::invalid_argument(std::string("Field '") + name + "' must be a non-empty string.");
    }
    const std::string value = static_cast<std::string>(node);
    if (value.empty()) {
        throw std::invalid_argument(std::string("Field '") + name + "' must be a non-empty string.");
    }
    return value;
}

float required_probability(const cv::FileNode& root, const char* name) {
    const cv::FileNode node = root[name];
    if (node.empty() || (!node.isReal() && !node.isInt())) {
        throw std::invalid_argument(std::string("Field '") + name + "' must be numeric.");
    }
    const double value = static_cast<double>(node);
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument(std::string("Field '") + name + "' must be between 0 and 1.");
    }
    return static_cast<float>(value);
}

WorkerRequest parse_request(const std::string& line) {
    cv::FileStorage storage;
    try {
        storage.open(line, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    } catch (const cv::Exception& exc) {
        throw std::invalid_argument(std::string("Invalid JSON: ") + exc.what());
    }
    if (!storage.isOpened()) {
        throw std::invalid_argument("Invalid JSON request.");
    }
    const cv::FileNode root = storage.root();
    if (!root.isMap()) {
        throw std::invalid_argument("JSON request root must be an object.");
    }

    WorkerRequest request;
    request.request_id = required_string(root, "request_id");
    request.command = required_string(root, "command");
    if (request.command == "shutdown") {
        return request;
    }
    if (request.command != "infer") {
        throw std::invalid_argument("Field 'command' must be 'infer' or 'shutdown'.");
    }

    request.image = required_string(root, "image");
    request.confidence = required_probability(root, "confidence");
    request.iou = required_probability(root, "iou");
    const cv::FileNode output = root["output"];
    if (!output.empty()) {
        if (!output.isString()) {
            throw std::invalid_argument("Field 'output' must be a string when provided.");
        }
        request.output = static_cast<std::string>(output);
    }
    return request;
}

void write_error(const std::string& request_id, const std::string& error_type, const std::string& message) {
    std::ostringstream out;
    out << "{\"request_id\":\"" << json_escape(request_id)
        << "\",\"ok\":false,\"error_type\":\"" << json_escape(error_type)
        << "\",\"message\":\"" << json_escape(message) << "\"}";
    write_protocol_line(out.str());
}

void draw_result_image(
    const fs::path& path,
    cv::Mat image,
    const std::vector<Detection>& detections
) {
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(40, 180, 255),
        cv::Scalar(80, 220, 90),
        cv::Scalar(230, 90, 120)
    };
    for (const Detection& detection : detections) {
        const cv::Scalar color = colors[
            static_cast<std::size_t>(std::max(0, detection.class_id)) % colors.size()
        ];
        const cv::Rect rect(
            static_cast<int>(std::round(detection.box.x)),
            static_cast<int>(std::round(detection.box.y)),
            static_cast<int>(std::round(detection.box.width)),
            static_cast<int>(std::round(detection.box.height))
        );
        cv::rectangle(image, rect, color, 2);
        std::ostringstream label;
        label << detection.class_name << ' '
              << std::fixed << std::setprecision(3) << detection.confidence;
        cv::putText(
            image,
            label.str(),
            cv::Point(rect.x, std::max(16, rect.y - 6)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv::LINE_AA
        );
    }
    if (!write_image_unicode(path, image)) {
        throw std::runtime_error("Failed to write result image: " + path_to_utf8(path));
    }
}

void write_infer_response(
    const WorkerRequest& request,
    const std::string& engine_label,
    const InferenceResult& result,
    double ipc_worker_ms
) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\"request_id\":\"" << json_escape(request.request_id)
        << "\",\"ok\":true,\"backend\":\"tensorrt\",\"engine_label\":\""
        << json_escape(engine_label) << "\",\"detections\":[";
    for (std::size_t index = 0; index < result.detections.size(); ++index) {
        const Detection& detection = result.detections[index];
        if (index > 0) {
            out << ',';
        }
        out << "{\"class_id\":" << detection.class_id
            << ",\"class_name\":\"" << json_escape(detection.class_name)
            << "\",\"confidence\":" << detection.confidence
            << ",\"bbox\":[" << detection.box.x << ',' << detection.box.y << ','
            << detection.box.x + detection.box.width << ','
            << detection.box.y + detection.box.height << "]}";
    }
    out << "],\"timing_ms\":{\"preprocess\":" << result.preprocess_ms
        << ",\"inference\":" << result.inference_ms
        << ",\"postprocess\":" << result.postprocess_ms
        << ",\"total\":" << result.total_ms
        << ",\"worker_request\":" << ipc_worker_ms << "}}";
    write_protocol_line(out.str());
}

}  // namespace

int run_tensorrt_worker(
    const std::string& engine_path,
    const std::vector<std::string>& class_names,
    const std::string& engine_label,
    int image_size,
    int device_id
) {
    const auto startup_start = std::chrono::steady_clock::now();
    try {
        TensorRtDetector detector(engine_path, class_names, image_size, device_id);
        std::ostringstream ready;
        ready << std::fixed << std::setprecision(6)
              << "{\"event\":\"ready\",\"ok\":true,\"backend\":\"tensorrt\","
              << "\"engine_label\":\"" << json_escape(engine_label)
              << "\",\"device_id\":" << device_id
              << ",\"startup_ms\":" << elapsed_ms(startup_start) << "}";
        write_protocol_line(ready.str());
        std::cerr << "TensorRT worker started\n";

        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }

            const std::string fallback_request_id = request_id_from_line(line);
            try {
                const WorkerRequest request = parse_request(line);
                if (request.command == "shutdown") {
                    write_protocol_line(
                        "{\"request_id\":\"" + json_escape(request.request_id)
                        + "\",\"ok\":true,\"status\":\"shutdown\"}"
                    );
                    return 0;
                }

                const fs::path image_path = path_from_utf8(request.image);
                if (!fs::is_regular_file(image_path)) {
                    throw std::invalid_argument("Image file not found: " + request.image);
                }
                const cv::Mat image = load_bgr_image(image_path);
                const auto request_start = std::chrono::steady_clock::now();
                const InferenceResult result = detector.infer(image, request.confidence, request.iou);
                if (!request.output.empty()) {
                    const fs::path output_dir = path_from_utf8(request.output);
                    fs::create_directories(output_dir);
                    draw_result_image(output_dir / "result.jpg", image.clone(), result.detections);
                }
                write_infer_response(request, engine_label, result, elapsed_ms(request_start));
            } catch (const std::invalid_argument& exc) {
                write_error(fallback_request_id, "InvalidRequest", exc.what());
            } catch (const cv::Exception& exc) {
                write_error(fallback_request_id, "ImageLoadError", exc.what());
            } catch (const std::exception& exc) {
                write_error(fallback_request_id, "InferenceError", exc.what());
            }
        }
        return 0;
    } catch (const std::exception& exc) {
        write_protocol_line(
            "{\"event\":\"ready\",\"ok\":false,\"message\":\""
            + json_escape(exc.what()) + "\"}"
        );
        std::cerr << "TensorRT worker startup failed: " << exc.what() << '\n';
        return 2;
    }
}

}  // namespace pcb_vision
