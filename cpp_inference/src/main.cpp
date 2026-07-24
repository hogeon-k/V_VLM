#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
};

void print_usage(const char* program_name) {
    std::cout
        << "Usage: " << program_name << " --model <best.onnx> --metadata <model_metadata.json> "
        << "--image <image> --output <dir> [--imgsz 960] [--conf 0.15] [--iou 0.7]\n";
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
    out << "  \"config\": {\"imgsz\": " << args.imgsz << ", \"conf\": " << args.conf << ", \"iou\": " << args.iou << "},\n";
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
        const std::vector<std::string> class_names = load_class_names(args.metadata);
        const cv::Mat image = pcb_vision::load_bgr_image(args.image);
        fs::create_directories(args.output);

        pcb_vision::OnnxDetector detector(args.model, class_names, args.imgsz);
        pcb_vision::InferenceResult result = detector.infer(image, args.conf, args.iou);

        const fs::path output_dir(args.output);
        write_json(output_dir / "result.json", args, result);
        write_csv(output_dir / "detections.csv", result.detections);
        draw_result_image(output_dir / "result.jpg", image.clone(), result.detections);

        std::cout << "=== C++ ONNX Runtime Inference ===\n";
        std::cout << "Model: " << args.model << '\n';
        std::cout << "Image: " << args.image << '\n';
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
        std::cout << "Preprocess: " << result.preprocess_ms << " ms\n";
        std::cout << "Inference: " << result.inference_ms << " ms\n";
        std::cout << "Postprocess: " << result.postprocess_ms << " ms\n";
        std::cout << "Total: " << result.total_ms << " ms\n";
        std::cout << "Output: " << output_dir.string() << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << '\n';
        return 1;
    }
}
