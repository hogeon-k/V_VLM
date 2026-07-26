#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "batch_benchmark.hpp"
#include "image_preprocessor.hpp"
#include "tensorrt_detector.hpp"

namespace fs = std::filesystem;
namespace bench = pcb_vision::benchmark;

namespace {

struct Args {
    std::string engine;
    std::string engine_label;
    std::string metadata = "models/model_metadata.json";
    std::string images;
    std::string output;
    int device_id = 0;
    int imgsz = 960;
    float conf = 0.15F;
    float iou = 0.7F;
    double match_iou = 0.5;
    int warmup = 10;
    int repeat = 30;
    bool recursive = false;
    std::string extensions = "jpg,jpeg,png,bmp";
    int max_images = 0;
    unsigned int seed = 0;
    bool use_seed = false;
    double strict_confidence_tolerance = 0.001;
    double practical_confidence_tolerance = 0.002;
    double bbox_tolerance = 1.0;
    bool fail_on_mismatch = false;
};

struct RunTimings {
    std::vector<double> h2d_ms;
    std::vector<double> gpu_execution_ms;
    std::vector<double> d2h_ms;
    std::vector<double> inference_total_ms;
    std::vector<double> postprocess_ms;
    std::vector<double> end_to_end_ms;
    int validation_mismatches = 0;
    int numerical_warnings = 0;
    pcb_vision::InferenceResult baseline;
};

struct ImageReport {
    fs::path image_path;
    int width = 0;
    int height = 0;
    int detection_count = 0;
    int validation_mismatches = 0;
    int numerical_warnings = 0;
    std::string status = "PASS";
    std::string failure_reason = "within tolerances";
    std::string error;
    double preprocess_ms = 0.0;
    bench::TimingStats h2d;
    bench::TimingStats gpu_execution;
    bench::TimingStats d2h;
    bench::TimingStats inference_total;
    bench::TimingStats postprocess;
    bench::TimingStats end_to_end;
    std::vector<pcb_vision::Detection> detections;
};

double elapsed_ms(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::string require_value(int& index, int argc, char* argv[], const std::string& option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(option + " requires a value.");
    }
    return argv[++index];
}

void print_usage(const char* exe) {
    std::cout
        << "Usage: " << exe << " --engine <best.engine> --engine-label fp32|fp16 "
        << "--images <dir> --output <dir> [--metadata models/model_metadata.json]\n"
        << "       [--device-id 0] [--imgsz 960] [--conf 0.15] [--iou 0.7] "
        << "[--match-iou 0.5] [--warmup 10] [--repeat 30]\n"
        << "       [--strict-confidence-tolerance 0.001] "
        << "[--practical-confidence-tolerance 0.002] [--bbox-tolerance 1.0]\n";
}

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (option == "--engine") {
            args.engine = require_value(index, argc, argv, option);
        } else if (option == "--engine-label") {
            args.engine_label = require_value(index, argc, argv, option);
            std::transform(args.engine_label.begin(), args.engine_label.end(), args.engine_label.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });
            if (args.engine_label != "fp32" && args.engine_label != "fp16") {
                throw std::invalid_argument("--engine-label must be fp32 or fp16.");
            }
        } else if (option == "--metadata") {
            args.metadata = require_value(index, argc, argv, option);
        } else if (option == "--images") {
            args.images = require_value(index, argc, argv, option);
        } else if (option == "--output") {
            args.output = require_value(index, argc, argv, option);
        } else if (option == "--device-id") {
            args.device_id = std::stoi(require_value(index, argc, argv, option));
        } else if (option == "--imgsz") {
            args.imgsz = std::stoi(require_value(index, argc, argv, option));
        } else if (option == "--conf") {
            args.conf = std::stof(require_value(index, argc, argv, option));
        } else if (option == "--iou") {
            args.iou = std::stof(require_value(index, argc, argv, option));
        } else if (option == "--match-iou") {
            args.match_iou = std::stod(require_value(index, argc, argv, option));
        } else if (option == "--warmup") {
            args.warmup = std::stoi(require_value(index, argc, argv, option));
        } else if (option == "--repeat") {
            args.repeat = std::stoi(require_value(index, argc, argv, option));
        } else if (option == "--recursive") {
            args.recursive = true;
        } else if (option == "--extensions") {
            args.extensions = require_value(index, argc, argv, option);
        } else if (option == "--max-images") {
            args.max_images = std::stoi(require_value(index, argc, argv, option));
        } else if (option == "--seed") {
            args.seed = static_cast<unsigned int>(std::stoul(require_value(index, argc, argv, option)));
            args.use_seed = true;
        } else if (option == "--strict-confidence-tolerance") {
            args.strict_confidence_tolerance = std::stod(require_value(index, argc, argv, option));
        } else if (option == "--practical-confidence-tolerance") {
            args.practical_confidence_tolerance = std::stod(require_value(index, argc, argv, option));
        } else if (option == "--bbox-tolerance") {
            args.bbox_tolerance = std::stod(require_value(index, argc, argv, option));
        } else if (option == "--fail-on-mismatch") {
            args.fail_on_mismatch = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + option);
        }
    }
    if (args.engine.empty() || args.engine_label.empty() || args.images.empty() || args.output.empty()) {
        throw std::invalid_argument("--engine, --engine-label, --images, and --output are required.");
    }
    if (args.repeat <= 0 || args.warmup < 0 || args.imgsz <= 0) {
        throw std::invalid_argument("--repeat must be > 0, --warmup must be >= 0, and --imgsz must be > 0.");
    }
    bench::validate_confidence_tolerances(
        args.strict_confidence_tolerance,
        args.practical_confidence_tolerance
    );
    return args;
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

std::string repeat_status(
    const std::vector<pcb_vision::Detection>& baseline,
    const std::vector<pcb_vision::Detection>& current,
    const Args& args
) {
    const bench::MatchResult comparison = bench::compare_detections(baseline, current, args.match_iou);
    if (bench::has_structural_mismatch(
            static_cast<int>(baseline.size()),
            static_cast<int>(current.size()),
            comparison
        )
        || comparison.max_confidence_diff > args.practical_confidence_tolerance
        || comparison.max_bbox_diff > args.bbox_tolerance) {
        return "FAIL";
    }
    if (comparison.max_confidence_diff > args.strict_confidence_tolerance) {
        return "NUMERICAL_WARNING";
    }
    return "PASS";
}

RunTimings run_repeated(
    pcb_vision::TensorRtDetector& detector,
    pcb_vision::PreprocessResult& preprocess,
    cv::Size original_size,
    const Args& args,
    double preprocess_ms
) {
    RunTimings timings;
    bool has_baseline = false;
    for (int repeat = 0; repeat < args.repeat; ++repeat) {
        pcb_vision::InferenceResult result = detector.infer_preprocessed(preprocess, original_size, args.conf, args.iou);
        const pcb_vision::TensorRtRunTiming timing = detector.last_timing();
        if (!has_baseline) {
            timings.baseline = result;
            has_baseline = true;
        } else {
            const std::string status = repeat_status(timings.baseline.detections, result.detections, args);
            if (status != "PASS") {
                ++timings.validation_mismatches;
                if (status == "NUMERICAL_WARNING") {
                    ++timings.numerical_warnings;
                }
            }
        }
        timings.h2d_ms.push_back(timing.h2d_ms);
        timings.gpu_execution_ms.push_back(timing.gpu_execution_ms);
        timings.d2h_ms.push_back(timing.d2h_ms);
        timings.inference_total_ms.push_back(timing.total_ms);
        timings.postprocess_ms.push_back(result.postprocess_ms);
        timings.end_to_end_ms.push_back(preprocess_ms + timing.total_ms + result.postprocess_ms);
    }
    return timings;
}

void write_stats_json(std::ostream& out, const std::string& key, const bench::TimingStats& stats, const std::string& indent) {
    out << indent << "\"" << key << "\": {"
        << "\"count\": " << stats.count
        << ", \"first\": " << stats.first
        << ", \"min\": " << stats.min
        << ", \"mean\": " << stats.mean
        << ", \"median\": " << stats.median
        << ", \"p95\": " << stats.p95
        << ", \"max\": " << stats.max
        << ", \"stddev\": " << stats.stddev
        << "}";
}

void append_values(std::vector<double>& target, const std::vector<double>& values) {
    target.insert(target.end(), values.begin(), values.end());
}

bench::TimingStats stats_from_reports(const std::vector<ImageReport>& reports, const std::string& key) {
    std::vector<double> values;
    for (const ImageReport& report : reports) {
        if (report.status == "ERROR") {
            continue;
        }
        if (key == "preprocess") {
            values.push_back(report.preprocess_ms);
        } else if (key == "h2d") {
            values.push_back(report.h2d.mean);
        } else if (key == "gpu") {
            values.push_back(report.gpu_execution.mean);
        } else if (key == "d2h") {
            values.push_back(report.d2h.mean);
        } else if (key == "inference_total") {
            values.push_back(report.inference_total.mean);
        } else if (key == "postprocess") {
            values.push_back(report.postprocess.mean);
        } else if (key == "end_to_end") {
            values.push_back(report.end_to_end.mean);
        }
    }
    return bench::calculate_stats(values);
}

void write_summary_json(
    const fs::path& path,
    const Args& args,
    const pcb_vision::TensorRtDetector& detector,
    const std::vector<ImageReport>& reports
) {
    const int failed_image_count = static_cast<int>(std::count_if(
        reports.begin(), reports.end(), [](const ImageReport& report) { return report.status == "ERROR"; }
    ));
    const int validation_mismatch_count = std::accumulate(
        reports.begin(), reports.end(), 0, [](int sum, const ImageReport& report) {
            return sum + report.validation_mismatches;
        }
    );
    const int total_detection_count = std::accumulate(
        reports.begin(), reports.end(), 0, [](int sum, const ImageReport& report) {
            return sum + report.detection_count;
        }
    );
    const bench::TimingStats preprocess = stats_from_reports(reports, "preprocess");
    const bench::TimingStats h2d = stats_from_reports(reports, "h2d");
    const bench::TimingStats gpu = stats_from_reports(reports, "gpu");
    const bench::TimingStats d2h = stats_from_reports(reports, "d2h");
    const bench::TimingStats inference_total = stats_from_reports(reports, "inference_total");
    const bench::TimingStats postprocess = stats_from_reports(reports, "postprocess");
    const bench::TimingStats end_to_end = stats_from_reports(reports, "end_to_end");
    const double qps = end_to_end.mean <= 0.0 ? 0.0 : 1000.0 / end_to_end.mean;

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write summary JSON: " + path.string());
    }
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"backend\": \"tensorrt\",\n";
    out << "  \"engine_path\": \"" << json_escape(args.engine) << "\",\n";
    out << "  \"engine_label\": \"" << json_escape(args.engine_label) << "\",\n";
    out << "  \"tensorrt_version\": \"" << detector.version_string() << "\",\n";
    out << "  \"device_id\": " << args.device_id << ",\n";
    out << "  \"image_count\": " << reports.size() << ",\n";
    out << "  \"total_detection_count\": " << total_detection_count << ",\n";
    out << "  \"warmup\": " << args.warmup << ",\n";
    out << "  \"repeat\": " << args.repeat << ",\n";
    out << "  \"input\": {\"name\": \"" << json_escape(detector.input_info().name)
        << "\", \"dtype\": \"" << json_escape(detector.input_info().dtype) << "\"},\n";
    out << "  \"output\": {\"name\": \"" << json_escape(detector.output_info().name)
        << "\", \"dtype\": \"" << json_escape(detector.output_info().dtype) << "\"},\n";
    out << "  \"timing\": {\n";
    write_stats_json(out, "gpu_execution", gpu, "    ");
    out << ",\n";
    write_stats_json(out, "h2d", h2d, "    ");
    out << ",\n";
    write_stats_json(out, "d2h", d2h, "    ");
    out << ",\n";
    write_stats_json(out, "inference_total", inference_total, "    ");
    out << ",\n";
    write_stats_json(out, "preprocess", preprocess, "    ");
    out << ",\n";
    write_stats_json(out, "postprocess", postprocess, "    ");
    out << ",\n";
    write_stats_json(out, "end_to_end", end_to_end, "    ");
    out << "\n  },\n";
    out << "  \"qps\": " << qps << ",\n";
    out << "  \"validation_mismatch_count\": " << validation_mismatch_count << ",\n";
    out << "  \"failed_image_count\": " << failed_image_count << "\n";
    out << "}\n";
}

void write_per_image_csv(const fs::path& path, const Args& args, const std::vector<ImageReport>& reports) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write per-image CSV: " + path.string());
    }
    out << "image_name,engine_label,detection_count,detection_index,class_id,class_name,confidence,"
        << "x1,y1,x2,y2,preprocess_ms,h2d_mean_ms,gpu_execution_mean_ms,d2h_mean_ms,"
        << "inference_total_mean_ms,postprocess_mean_ms,end_to_end_mean_ms,validation_mismatches,status,error\n";
    out << std::fixed << std::setprecision(6);
    for (const ImageReport& report : reports) {
        const auto write_common = [&](int detection_index, const pcb_vision::Detection* detection) {
            out << report.image_path.filename().string() << ','
                << args.engine_label << ','
                << report.detection_count << ','
                << detection_index << ',';
            if (detection == nullptr) {
                out << ",,,,,,";
            } else {
                out << detection->class_id << ','
                    << detection->class_name << ','
                    << detection->confidence << ','
                    << detection->box.x << ','
                    << detection->box.y << ','
                    << detection->box.x + detection->box.width << ','
                    << detection->box.y + detection->box.height;
            }
            out << ',' << report.preprocess_ms
                << ',' << report.h2d.mean
                << ',' << report.gpu_execution.mean
                << ',' << report.d2h.mean
                << ',' << report.inference_total.mean
                << ',' << report.postprocess.mean
                << ',' << report.end_to_end.mean
                << ',' << report.validation_mismatches
                << ',' << report.status
                << ",\"" << json_escape(report.error) << "\"\n";
        };
        if (report.detections.empty()) {
            write_common(-1, nullptr);
        } else {
            for (std::size_t index = 0; index < report.detections.size(); ++index) {
                write_common(static_cast<int>(index), &report.detections[index]);
            }
        }
    }
}

void write_detections_json(const fs::path& path, const Args& args, const std::vector<ImageReport>& reports) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write detections JSON: " + path.string());
    }
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"backend\": \"tensorrt\",\n";
    out << "  \"engine_label\": \"" << json_escape(args.engine_label) << "\",\n";
    out << "  \"images\": [\n";
    for (std::size_t image_index = 0; image_index < reports.size(); ++image_index) {
        const ImageReport& report = reports[image_index];
        out << "    {\"image\": \"" << json_escape(report.image_path.string()) << "\", "
            << "\"status\": \"" << report.status << "\", \"detections\": [";
        for (std::size_t det_index = 0; det_index < report.detections.size(); ++det_index) {
            const auto& det = report.detections[det_index];
            out << (det_index > 0 ? ", " : "")
                << "{\"class_id\": " << det.class_id
                << ", \"class_name\": \"" << json_escape(det.class_name) << "\""
                << ", \"confidence\": " << det.confidence
                << ", \"bbox\": [" << det.box.x << ", " << det.box.y << ", "
                << det.box.x + det.box.width << ", " << det.box.y + det.box.height << "]}";
        }
        out << "]}";
        out << (image_index + 1 < reports.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

void create_failure_keep_file(const fs::path& output) {
    const fs::path failure_dir = output / "failure_cases";
    fs::create_directories(failure_dir);
    std::ofstream keep(failure_dir / ".gitkeep");
    if (!keep) {
        throw std::runtime_error("Failed to write failure_cases/.gitkeep under: " + output.string());
    }
}

std::string final_status(const std::vector<ImageReport>& reports) {
    if (std::any_of(reports.begin(), reports.end(), [](const ImageReport& report) {
            return report.status == "ERROR" || report.status == "FAIL";
        })) {
        return "FAIL";
    }
    if (std::any_of(reports.begin(), reports.end(), [](const ImageReport& report) {
            return report.status == "NUMERICAL_WARNING";
        })) {
        return "NUMERICAL_WARNING";
    }
    return "PASS";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const Args args = parse_args(argc, argv);
        const fs::path output_dir(args.output);
        fs::create_directories(output_dir);
        create_failure_keep_file(output_dir);

        std::vector<fs::path> images = bench::collect_images(args.images, bench::parse_extensions(args.extensions), args.recursive);
        if (args.use_seed) {
            std::mt19937 rng(args.seed);
            std::shuffle(images.begin(), images.end(), rng);
        }
        if (args.max_images > 0 && static_cast<std::size_t>(args.max_images) < images.size()) {
            images.resize(static_cast<std::size_t>(args.max_images));
        }
        if (images.empty()) {
            throw std::runtime_error("No supported images found under: " + args.images);
        }

        const std::vector<std::string> class_names = load_class_names(args.metadata);
        pcb_vision::TensorRtDetector detector(args.engine, class_names, args.imgsz, args.device_id);
        std::vector<ImageReport> reports;
        reports.reserve(images.size());

        for (std::size_t index = 0; index < images.size(); ++index) {
            ImageReport report;
            report.image_path = images[index];
            try {
                const auto e2e_start = std::chrono::steady_clock::now();
                const cv::Mat image = pcb_vision::load_bgr_image(images[index].string());
                report.width = image.cols;
                report.height = image.rows;
                const auto preprocess_start = std::chrono::steady_clock::now();
                pcb_vision::PreprocessResult preprocess = pcb_vision::preprocess_image(image, args.imgsz);
                report.preprocess_ms = elapsed_ms(preprocess_start);

                for (int warmup = 0; warmup < args.warmup; ++warmup) {
                    (void)detector.infer_preprocessed(preprocess, image.size(), args.conf, args.iou);
                }

                RunTimings timings = run_repeated(detector, preprocess, image.size(), args, report.preprocess_ms);
                report.detections = timings.baseline.detections;
                report.detection_count = static_cast<int>(report.detections.size());
                report.validation_mismatches = timings.validation_mismatches;
                report.numerical_warnings = timings.numerical_warnings;
                report.h2d = bench::calculate_stats(timings.h2d_ms);
                report.gpu_execution = bench::calculate_stats(timings.gpu_execution_ms);
                report.d2h = bench::calculate_stats(timings.d2h_ms);
                report.inference_total = bench::calculate_stats(timings.inference_total_ms);
                report.postprocess = bench::calculate_stats(timings.postprocess_ms);
                report.end_to_end = bench::calculate_stats(timings.end_to_end_ms);
                if (report.validation_mismatches > timings.numerical_warnings) {
                    report.status = "FAIL";
                    report.failure_reason = "repeat validation mismatch";
                } else if (report.numerical_warnings > 0) {
                    report.status = "NUMERICAL_WARNING";
                    report.failure_reason = "repeat numerical difference";
                }
                const double wall_e2e_ms = elapsed_ms(e2e_start);
                (void)wall_e2e_ms;
            } catch (const std::exception& exc) {
                report.status = "ERROR";
                report.failure_reason = "image processing failed";
                report.error = exc.what();
            }
            reports.push_back(report);
            std::cout << "[" << index + 1 << "/" << images.size() << "] "
                      << images[index].filename().string() << ' ' << report.status
                      << " detections=" << report.detection_count
                      << " gpu_mean_ms=" << report.gpu_execution.mean
                      << " e2e_mean_ms=" << report.end_to_end.mean << '\n';
        }

        write_summary_json(output_dir / "summary.json", args, detector, reports);
        write_per_image_csv(output_dir / "per_image.csv", args, reports);
        write_detections_json(output_dir / "detections.json", args, reports);

        const int total_detection_count = std::accumulate(
            reports.begin(), reports.end(), 0, [](int sum, const ImageReport& report) {
                return sum + report.detection_count;
            }
        );
        const int validation_mismatch_count = std::accumulate(
            reports.begin(), reports.end(), 0, [](int sum, const ImageReport& report) {
                return sum + report.validation_mismatches;
            }
        );
        const int failed_image_count = static_cast<int>(std::count_if(
            reports.begin(), reports.end(), [](const ImageReport& report) { return report.status == "ERROR"; }
        ));
        const bench::TimingStats gpu = stats_from_reports(reports, "gpu");
        const bench::TimingStats h2d = stats_from_reports(reports, "h2d");
        const bench::TimingStats d2h = stats_from_reports(reports, "d2h");
        const bench::TimingStats inference_total = stats_from_reports(reports, "inference_total");
        const bench::TimingStats preprocess = stats_from_reports(reports, "preprocess");
        const bench::TimingStats postprocess = stats_from_reports(reports, "postprocess");
        const bench::TimingStats end_to_end = stats_from_reports(reports, "end_to_end");
        const double qps = end_to_end.mean <= 0.0 ? 0.0 : 1000.0 / end_to_end.mean;
        const std::string status = final_status(reports);

        std::cout << "=== Native TensorRT Batch Benchmark ===\n";
        std::cout << "Engine path: " << args.engine << '\n';
        std::cout << "Engine label: " << args.engine_label << '\n';
        std::cout << "TensorRT version: " << detector.version_string() << '\n';
        std::cout << "CUDA device id: " << detector.device_id() << '\n';
        std::cout << "Image count: " << reports.size() << '\n';
        std::cout << "Total detection count: " << total_detection_count << '\n';
        std::cout << "Warmup: " << args.warmup << '\n';
        std::cout << "Repeat: " << args.repeat << "\n\n";
        std::cout << "GPU execution mean: " << gpu.mean << '\n';
        std::cout << "GPU execution median: " << gpu.median << '\n';
        std::cout << "GPU execution p95: " << gpu.p95 << "\n\n";
        std::cout << "H2D mean: " << h2d.mean << '\n';
        std::cout << "D2H mean: " << d2h.mean << '\n';
        std::cout << "Inference total mean: " << inference_total.mean << "\n\n";
        std::cout << "Preprocess mean: " << preprocess.mean << '\n';
        std::cout << "Postprocess mean: " << postprocess.mean << '\n';
        std::cout << "End-to-end mean: " << end_to_end.mean << '\n';
        std::cout << "End-to-end median: " << end_to_end.median << '\n';
        std::cout << "End-to-end p95: " << end_to_end.p95 << '\n';
        std::cout << "QPS: " << qps << "\n\n";
        std::cout << "Validation mismatch count: " << validation_mismatch_count << '\n';
        std::cout << "Failed image count: " << failed_image_count << '\n';
        std::cout << "Final status: " << status << '\n';
        std::cout << "Output: " << output_dir.string() << '\n';
        return (args.fail_on_mismatch && status != "PASS") ? 2 : 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << '\n';
        return 1;
    }
}
