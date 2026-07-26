#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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
#include "detector.hpp"
#include "image_preprocessor.hpp"
#include "unicode_utils.hpp"

namespace fs = std::filesystem;
namespace bench = pcb_vision::benchmark;

namespace {

struct Args {
    std::string model;
    std::string metadata = "models/model_metadata.json";
    std::string images;
    std::string output;
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
    int device_id = 0;
    double strict_confidence_tolerance = 0.001;
    double practical_confidence_tolerance = 0.002;
    double bbox_tolerance = 1.0;
    bool fail_on_mismatch = false;
    std::string cudnn_conv_algo_search = "HEURISTIC";
    std::string provider_order = "alternate";
};

struct RunTimings {
    std::vector<double> session_run_ms;
    std::vector<double> postprocess_ms;
    std::vector<double> total_ms;
    int internal_mismatches = 0;
    pcb_vision::InferenceResult baseline;
};

struct ImageReport {
    fs::path image_path;
    int width = 0;
    int height = 0;
    bench::MatchResult comparison;
    int cpu_detection_count = 0;
    int cuda_detection_count = 0;
    bench::TimingStats cpu_session;
    bench::TimingStats cuda_session;
    bench::TimingStats cpu_total;
    bench::TimingStats cuda_total;
    std::vector<double> cpu_session_runs;
    std::vector<double> cuda_session_runs;
    std::vector<double> cpu_total_runs;
    std::vector<double> cuda_total_runs;
    int cpu_internal_mismatches = 0;
    int cuda_internal_mismatches = 0;
    std::string status;
    std::string failure_reason;
};

double elapsed_ms(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::string env_value(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) {
        return "unavailable";
    }
    std::string result(value);
    std::free(value);
    return result.empty() ? "unavailable" : result;
#else
    const char* value = std::getenv(name);
    return value == nullptr || std::string(value).empty() ? "unavailable" : std::string(value);
#endif
}

std::string require_value(int& index, const std::vector<std::string>& argv, const std::string& option) {
    if (index + 1 >= static_cast<int>(argv.size())) {
        throw std::invalid_argument(option + " requires a value.");
    }
    return argv[++index];
}

void print_usage(const std::string& exe) {
    std::cout
        << "Usage: " << exe << " --model <best.onnx> --images <dir> --output <dir> "
        << "[--metadata models/model_metadata.json] [--imgsz 960] [--conf 0.15] [--iou 0.7]\n"
        << "       [--match-iou 0.5] [--warmup 10] [--repeat 30] "
        << "[--strict-confidence-tolerance 0.001] [--practical-confidence-tolerance 0.002] "
        << "[--bbox-tolerance 1.0]\n"
        << "       "
        << "[--cudnn-conv-algo-search heuristic|exhaustive|default]\n"
        << "  --cudnn-conv-algo-search: allowed values: heuristic, exhaustive, default; "
        << "default: heuristic\n";
}

Args parse_args(const std::vector<std::string>& argv) {
    Args args;
    for (int index = 1; index < static_cast<int>(argv.size()); ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (option == "--model") {
            args.model = require_value(index, argv, option);
        } else if (option == "--metadata") {
            args.metadata = require_value(index, argv, option);
        } else if (option == "--images") {
            args.images = require_value(index, argv, option);
        } else if (option == "--output") {
            args.output = require_value(index, argv, option);
        } else if (option == "--imgsz") {
            args.imgsz = std::stoi(require_value(index, argv, option));
        } else if (option == "--conf") {
            args.conf = std::stof(require_value(index, argv, option));
        } else if (option == "--iou") {
            args.iou = std::stof(require_value(index, argv, option));
        } else if (option == "--match-iou") {
            args.match_iou = std::stod(require_value(index, argv, option));
        } else if (option == "--warmup") {
            args.warmup = std::stoi(require_value(index, argv, option));
        } else if (option == "--repeat") {
            args.repeat = std::stoi(require_value(index, argv, option));
        } else if (option == "--recursive") {
            args.recursive = true;
        } else if (option == "--extensions") {
            args.extensions = require_value(index, argv, option);
        } else if (option == "--max-images") {
            args.max_images = std::stoi(require_value(index, argv, option));
        } else if (option == "--seed") {
            args.seed = static_cast<unsigned int>(std::stoul(require_value(index, argv, option)));
            args.use_seed = true;
        } else if (option == "--device-id") {
            args.device_id = std::stoi(require_value(index, argv, option));
        } else if (option == "--confidence-tolerance") {
            const double legacy_tolerance = std::stod(require_value(index, argv, option));
            args.strict_confidence_tolerance = legacy_tolerance;
            args.practical_confidence_tolerance = legacy_tolerance;
        } else if (option == "--strict-confidence-tolerance") {
            args.strict_confidence_tolerance = std::stod(require_value(index, argv, option));
        } else if (option == "--practical-confidence-tolerance") {
            args.practical_confidence_tolerance = std::stod(require_value(index, argv, option));
        } else if (option == "--bbox-tolerance") {
            args.bbox_tolerance = std::stod(require_value(index, argv, option));
        } else if (option == "--fail-on-mismatch") {
            args.fail_on_mismatch = true;
        } else if (option == "--cudnn-conv-algo-search") {
            const std::string value = index + 1 < static_cast<int>(argv.size()) ? argv[++index] : "";
            args.cudnn_conv_algo_search = pcb_vision::normalize_cudnn_conv_algo_search(
                value
            );
        } else if (option == "--provider-order") {
            args.provider_order = require_value(index, argv, option);
            if (args.provider_order != "cpu-first" && args.provider_order != "cuda-first" && args.provider_order != "alternate") {
                throw std::invalid_argument("--provider-order must be cpu-first, cuda-first, or alternate.");
            }
        } else {
            throw std::invalid_argument("Unknown argument: " + option);
        }
    }
    if (args.model.empty() || args.images.empty() || args.output.empty()) {
        throw std::invalid_argument("--model, --images, and --output are required.");
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
    return pcb_vision::read_text_file_utf8(pcb_vision::path_from_utf8(path));
}

std::vector<std::string> fallback_class_names() {
    return {"open_circuit", "short", "missing_hole"};
}

std::vector<std::string> load_class_names(const std::string& metadata_path) {
    if (metadata_path.empty() || !fs::exists(pcb_vision::path_from_utf8(metadata_path))) {
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

bool detections_equal(
    const std::vector<pcb_vision::Detection>& expected,
    const std::vector<pcb_vision::Detection>& actual,
    double confidence_tolerance,
    double bbox_tolerance
) {
    const bench::MatchResult comparison = bench::compare_detections(expected, actual, 0.5);
    return expected.size() == actual.size()
        && comparison.cpu_only_count == 0
        && comparison.cuda_only_count == 0
        && comparison.max_confidence_diff <= confidence_tolerance
        && comparison.max_bbox_diff <= bbox_tolerance;
}

RunTimings run_repeated(
    pcb_vision::OnnxDetector& detector,
    pcb_vision::PreprocessResult& preprocess,
    cv::Size original_size,
    const Args& args,
    double preprocess_ms
) {
    RunTimings timings;
    bool has_baseline = false;
    for (int repeat = 0; repeat < args.repeat; ++repeat) {
        pcb_vision::InferenceResult result = detector.infer_preprocessed(preprocess, original_size, args.conf, args.iou);
        if (!has_baseline) {
            timings.baseline = result;
            has_baseline = true;
        } else if (!detections_equal(
            timings.baseline.detections,
            result.detections,
            args.strict_confidence_tolerance,
            args.bbox_tolerance
        )) {
            ++timings.internal_mismatches;
        }
        timings.session_run_ms.push_back(result.inference_ms);
        timings.postprocess_ms.push_back(result.postprocess_ms);
        timings.total_ms.push_back(preprocess_ms + result.inference_ms + result.postprocess_ms);
    }
    return timings;
}

void write_detection_json(const fs::path& path, const pcb_vision::InferenceResult& result) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to write prediction JSON: " + pcb_vision::path_to_utf8(path));
    }
    out << std::fixed << std::setprecision(6);
    out << "{\n  \"provider\": \"" << json_escape(result.provider) << "\",\n  \"detections\": [\n";
    for (std::size_t i = 0; i < result.detections.size(); ++i) {
        const auto& det = result.detections[i];
        out << "    {\"class_id\": " << det.class_id
            << ", \"class_name\": \"" << json_escape(det.class_name) << "\""
            << ", \"confidence\": " << det.confidence
            << ", \"bbox\": [" << det.box.x << ", " << det.box.y << ", "
            << det.box.x + det.box.width << ", " << det.box.y + det.box.height << "]}";
        out << (i + 1 < result.detections.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
}

void write_comparison_json(const fs::path& path, const ImageReport& report) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to write comparison JSON: " + pcb_vision::path_to_utf8(path));
    }
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"image\": \"" << json_escape(pcb_vision::path_to_utf8(report.image_path)) << "\",\n";
    out << "  \"status\": \"" << report.status << "\",\n";
    out << "  \"failure_reason\": \"" << json_escape(report.failure_reason) << "\",\n";
    out << "  \"cpu_detection_count\": " << report.cpu_detection_count << ",\n";
    out << "  \"cuda_detection_count\": " << report.cuda_detection_count << ",\n";
    out << "  \"matched_count\": " << report.comparison.matched_count << ",\n";
    out << "  \"cpu_only_count\": " << report.comparison.cpu_only_count << ",\n";
    out << "  \"cuda_only_count\": " << report.comparison.cuda_only_count << ",\n";
    out << "  \"class_mismatch_count\": " << report.comparison.class_mismatch_count << ",\n";
    out << "  \"avg_confidence_diff\": " << report.comparison.avg_confidence_diff << ",\n";
    out << "  \"max_confidence_diff\": " << report.comparison.max_confidence_diff << ",\n";
    out << "  \"avg_bbox_diff\": " << report.comparison.avg_bbox_diff << ",\n";
    out << "  \"max_bbox_diff\": " << report.comparison.max_bbox_diff << ",\n";
    out << "  \"avg_matched_iou\": " << report.comparison.avg_matched_iou << ",\n";
    out << "  \"min_matched_iou\": " << report.comparison.min_matched_iou << "\n";
    out << "}\n";
}

void write_timing_csv(const fs::path& path, const std::vector<ImageReport>& reports) {
    std::ofstream out(path, std::ios::binary);
    out << "image,provider,repeat_index,session_run_ms,total_ms\n";
    out << std::fixed << std::setprecision(6);
    for (const ImageReport& report : reports) {
        for (std::size_t index = 0; index < report.cpu_session_runs.size(); ++index) {
            out << pcb_vision::path_to_utf8(report.image_path.filename()) << ",cpu," << index << ','
                << report.cpu_session_runs[index] << ',' << report.cpu_total_runs[index] << '\n';
        }
        for (std::size_t index = 0; index < report.cuda_session_runs.size(); ++index) {
            out << pcb_vision::path_to_utf8(report.image_path.filename()) << ",cuda," << index << ','
                << report.cuda_session_runs[index] << ',' << report.cuda_total_runs[index] << '\n';
        }
    }
}

void write_image_results_csv(
    const fs::path& path,
    const Args& args,
    const std::vector<ImageReport>& reports
) {
    const int pass_count = static_cast<int>(std::count_if(
        reports.begin(), reports.end(), [](const ImageReport& item) { return item.status == "PASS"; }
    ));
    const int numerical_warning_count = static_cast<int>(std::count_if(
        reports.begin(), reports.end(), [](const ImageReport& item) { return item.status == "NUMERICAL_WARNING"; }
    ));
    const int fail_count = static_cast<int>(std::count_if(
        reports.begin(), reports.end(), [](const ImageReport& item) { return item.status == "FAIL"; }
    ));
    const int structural_mismatch_count = static_cast<int>(std::count_if(
        reports.begin(), reports.end(), [](const ImageReport& item) {
            return bench::has_structural_mismatch(
                item.cpu_detection_count,
                item.cuda_detection_count,
                item.comparison
            );
        }
    ));
    const auto max_confidence = std::max_element(
        reports.begin(), reports.end(), [](const ImageReport& left, const ImageReport& right) {
            return left.comparison.max_confidence_diff < right.comparison.max_confidence_diff;
        }
    );
    const auto max_bbox = std::max_element(
        reports.begin(), reports.end(), [](const ImageReport& left, const ImageReport& right) {
            return left.comparison.max_bbox_diff < right.comparison.max_bbox_diff;
        }
    );
    std::ofstream out(path, std::ios::binary);
    out << "image,width,height,cpu_detection_count,cuda_detection_count,matched_count,cpu_only_count,cuda_only_count,"
        << "class_mismatch_count,avg_confidence_diff,max_confidence_diff,avg_bbox_diff,max_bbox_diff,avg_matched_iou,"
        << "min_matched_iou,cpu_session_mean_ms,cpu_session_median_ms,cpu_session_p95_ms,cuda_session_mean_ms,"
        << "cuda_session_median_ms,cuda_session_p95_ms,session_speedup,cpu_total_mean_ms,cuda_total_mean_ms,"
        << "total_speedup,cpu_internal_mismatches,cuda_internal_mismatches,structural_detection_mismatch,"
        << "strict_confidence_tolerance,practical_confidence_tolerance,bbox_tolerance,match_iou_threshold,"
        << "summary_pass_count,summary_numerical_warning_count,summary_fail_count,"
        << "summary_structural_detection_mismatch_count,summary_max_confidence_diff,"
        << "summary_max_bbox_diff,status,failure_reason\n";
    out << std::fixed << std::setprecision(6);
    for (const ImageReport& report : reports) {
        out << pcb_vision::path_to_utf8(report.image_path.filename()) << ','
            << report.width << ',' << report.height << ','
            << report.cpu_detection_count << ',' << report.cuda_detection_count << ','
            << report.comparison.matched_count << ',' << report.comparison.cpu_only_count << ','
            << report.comparison.cuda_only_count << ',' << report.comparison.class_mismatch_count << ','
            << report.comparison.avg_confidence_diff << ',' << report.comparison.max_confidence_diff << ','
            << report.comparison.avg_bbox_diff << ',' << report.comparison.max_bbox_diff << ','
            << report.comparison.avg_matched_iou << ',' << report.comparison.min_matched_iou << ','
            << report.cpu_session.mean << ',' << report.cpu_session.median << ',' << report.cpu_session.p95 << ','
            << report.cuda_session.mean << ',' << report.cuda_session.median << ',' << report.cuda_session.p95 << ','
            << bench::speedup(report.cpu_session.mean, report.cuda_session.mean) << ','
            << report.cpu_total.mean << ',' << report.cuda_total.mean << ','
            << bench::speedup(report.cpu_total.mean, report.cuda_total.mean) << ','
            << report.cpu_internal_mismatches << ',' << report.cuda_internal_mismatches << ','
            << (bench::has_structural_mismatch(
                    report.cpu_detection_count,
                    report.cuda_detection_count,
                    report.comparison
                ) ? 1 : 0) << ','
            << args.strict_confidence_tolerance << ','
            << args.practical_confidence_tolerance << ','
            << args.bbox_tolerance << ','
            << args.match_iou << ','
            << pass_count << ',' << numerical_warning_count << ',' << fail_count << ','
            << structural_mismatch_count << ','
            << (max_confidence == reports.end() ? 0.0 : max_confidence->comparison.max_confidence_diff) << ','
            << (max_bbox == reports.end() ? 0.0 : max_bbox->comparison.max_bbox_diff) << ','
            << report.status << ",\"" << report.failure_reason << "\"\n";
    }
}

bench::TimingStats merged_stats(const std::vector<ImageReport>& reports, bool cuda, bool total) {
    std::vector<double> values;
    for (const ImageReport& report : reports) {
        const bench::TimingStats& stats = cuda ? (total ? report.cuda_total : report.cuda_session) : (total ? report.cpu_total : report.cpu_session);
        values.push_back(stats.mean);
    }
    return bench::calculate_stats(values);
}

void write_environment_json(const fs::path& path, const Args& args) {
    std::ofstream out(path, std::ios::binary);
    out << "{\n";
    out << "  \"os\": \"Windows\",\n";
    out << "  \"cpu\": \"" << json_escape(env_value("PROCESSOR_IDENTIFIER")) << "\",\n";
    out << "  \"gpu\": \"unavailable\",\n";
    out << "  \"nvidia_driver\": \"unavailable\",\n";
    out << "  \"cuda_version\": \"12.6\",\n";
    out << "  \"cudnn_version\": \"9.25\",\n";
    out << "  \"onnxruntime_version\": \"unavailable\",\n";
    out << "  \"opencv_version\": \"" << CV_VERSION << "\",\n";
    out << "  \"msvc_compiler_version\": \"";
#ifdef _MSC_VER
    out << _MSC_VER;
#else
    out << "unavailable";
#endif
    out << "\",\n";
    out << "  \"cmake_version\": \"unavailable\",\n";
    out << "  \"model\": \"" << json_escape(args.model) << "\",\n";
    const fs::path model_path = pcb_vision::path_from_utf8(args.model);
    out << "  \"model_size\": " << (fs::exists(model_path) ? fs::file_size(model_path) : 0) << ",\n";
    out << "  \"model_sha256\": null\n";
    out << "}\n";
}

void write_summary_json(
    const fs::path& path,
    const Args& args,
    const std::vector<std::string>& available_providers,
    const std::vector<ImageReport>& reports
) {
    const int pass_count = static_cast<int>(std::count_if(reports.begin(), reports.end(), [](const ImageReport& item) { return item.status == "PASS"; }));
    const int numerical_warning_count = static_cast<int>(std::count_if(
        reports.begin(), reports.end(), [](const ImageReport& item) { return item.status == "NUMERICAL_WARNING"; }
    ));
    const int failed_count = static_cast<int>(std::count_if(reports.begin(), reports.end(), [](const ImageReport& item) { return item.status == "FAIL"; }));
    const int structural_mismatch_count = static_cast<int>(std::count_if(
        reports.begin(), reports.end(), [](const ImageReport& item) {
            return bench::has_structural_mismatch(
                item.cpu_detection_count,
                item.cuda_detection_count,
                item.comparison
            );
        }
    ));
    const int matched = std::accumulate(reports.begin(), reports.end(), 0, [](int sum, const ImageReport& item) { return sum + item.comparison.matched_count; });
    const int cpu_only = std::accumulate(reports.begin(), reports.end(), 0, [](int sum, const ImageReport& item) { return sum + item.comparison.cpu_only_count; });
    const int cuda_only = std::accumulate(reports.begin(), reports.end(), 0, [](int sum, const ImageReport& item) { return sum + item.comparison.cuda_only_count; });
    const int class_mismatches = std::accumulate(reports.begin(), reports.end(), 0, [](int sum, const ImageReport& item) { return sum + item.comparison.class_mismatch_count; });
    const int cpu_internal = std::accumulate(reports.begin(), reports.end(), 0, [](int sum, const ImageReport& item) { return sum + item.cpu_internal_mismatches; });
    const int cuda_internal = std::accumulate(reports.begin(), reports.end(), 0, [](int sum, const ImageReport& item) { return sum + item.cuda_internal_mismatches; });
    const auto max_conf = std::max_element(reports.begin(), reports.end(), [](const ImageReport& a, const ImageReport& b) {
        return a.comparison.max_confidence_diff < b.comparison.max_confidence_diff;
    });
    const auto max_bbox = std::max_element(reports.begin(), reports.end(), [](const ImageReport& a, const ImageReport& b) {
        return a.comparison.max_bbox_diff < b.comparison.max_bbox_diff;
    });
    const auto min_iou = std::min_element(reports.begin(), reports.end(), [](const ImageReport& a, const ImageReport& b) {
        return a.comparison.min_matched_iou < b.comparison.min_matched_iou;
    });
    const bench::TimingStats cpu_session = merged_stats(reports, false, false);
    const bench::TimingStats cuda_session = merged_stats(reports, true, false);
    const bench::TimingStats cpu_total = merged_stats(reports, false, true);
    const bench::TimingStats cuda_total = merged_stats(reports, true, true);
    const std::string final_status = failed_count > 0
        ? "FAIL"
        : (numerical_warning_count > 0 ? "NUMERICAL_WARNING" : "PASS");

    std::ofstream out(path, std::ios::binary);
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"config\": {\"model\": \"" << json_escape(args.model) << "\", \"metadata\": \"" << json_escape(args.metadata)
        << "\", \"images\": \"" << json_escape(args.images) << "\", \"image_count\": " << reports.size()
        << ", \"imgsz\": " << args.imgsz << ", \"conf\": " << args.conf << ", \"iou\": " << args.iou
        << ", \"match_iou\": " << args.match_iou << ", \"warmup\": " << args.warmup << ", \"repeat\": " << args.repeat
        << ", \"strict_confidence_tolerance\": " << args.strict_confidence_tolerance
        << ", \"practical_confidence_tolerance\": " << args.practical_confidence_tolerance
        << ", \"bbox_tolerance\": " << args.bbox_tolerance
        << ", \"cuda_device_id\": " << args.device_id << ", \"cudnn_conv_algo_search\": \"" << args.cudnn_conv_algo_search
        << "\", \"provider_order\": \"" << args.provider_order << "\"},\n";
    out << "  \"providers\": {\"available\": [";
    for (std::size_t i = 0; i < available_providers.size(); ++i) {
        out << (i > 0 ? ", " : "") << "\"" << available_providers[i] << "\"";
    }
    out << "], \"cpu\": {\"name\": \"CPUExecutionProvider\"}, \"cuda\": {\"name\": \"CUDAExecutionProvider\"}},\n";
    out << "  \"accuracy_comparison\": {\"passed_images\": " << pass_count
        << ", \"numerical_warning_images\": " << numerical_warning_count
        << ", \"warning_images\": " << numerical_warning_count
        << ", \"failed_images\": " << failed_count << ", \"matched_detections\": " << matched
        << ", \"cpu_only_detections\": " << cpu_only << ", \"cuda_only_detections\": " << cuda_only
        << ", \"class_mismatches\": " << class_mismatches
        << ", \"structural_detection_mismatch_images\": " << structural_mismatch_count
        << ", \"max_confidence_difference\": " << (max_conf == reports.end() ? 0.0 : max_conf->comparison.max_confidence_diff)
        << ", \"max_bbox_difference\": " << (max_bbox == reports.end() ? 0.0 : max_bbox->comparison.max_bbox_diff)
        << ", \"minimum_matched_iou\": " << (min_iou == reports.end() ? 1.0 : min_iou->comparison.min_matched_iou) << "},\n";
    out << "  \"timing\": {\"cpu\": {\"session_mean\": " << cpu_session.mean << ", \"session_median\": " << cpu_session.median
        << ", \"session_p95\": " << cpu_session.p95 << ", \"total_mean\": " << cpu_total.mean << ", \"total_median\": " << cpu_total.median
        << "}, \"cuda\": {\"session_mean\": " << cuda_session.mean << ", \"session_median\": " << cuda_session.median
        << ", \"session_p95\": " << cuda_session.p95 << ", \"total_mean\": " << cuda_total.mean << ", \"total_median\": " << cuda_total.median
        << "}, \"speedup\": {\"session_run_speedup_mean\": " << bench::speedup(cpu_session.mean, cuda_session.mean)
        << ", \"session_run_speedup_median\": " << bench::speedup(cpu_session.median, cuda_session.median)
        << ", \"session_run_speedup_p95\": " << bench::speedup(cpu_session.p95, cuda_session.p95)
        << ", \"end_to_end_speedup_mean\": " << bench::speedup(cpu_total.mean, cuda_total.mean)
        << ", \"end_to_end_speedup_median\": " << bench::speedup(cpu_total.median, cuda_total.median) << "}},\n";
    out << "  \"validation\": {\"cpu_internal_mismatches\": " << cpu_internal
        << ", \"cuda_internal_mismatches\": " << cuda_internal
        << ", \"cpu_cuda_mismatches\": " << (numerical_warning_count + failed_count) << "},\n";
    out << "  \"final_status\": \"" << final_status << "\"\n";
    out << "}\n";
}

void copy_failure(const fs::path& image, const fs::path& output, const std::string& reason) {
    fs::path category = "bbox_mismatch";
    if (reason.find("cpu_only") != std::string::npos) {
        category = "cpu_only";
    } else if (reason.find("cuda_only") != std::string::npos) {
        category = "cuda_only";
    } else if (reason.find("class_mismatch") != std::string::npos) {
        category = "class_mismatch";
    } else if (reason.find("confidence_mismatch") != std::string::npos) {
        category = "confidence_mismatch";
    } else if (reason.find("numerical_difference") != std::string::npos) {
        category = "numerical_difference";
    }
    fs::create_directories(output / "failure_cases" / category);
    fs::copy_file(image, output / "failure_cases" / category / image.filename(), fs::copy_options::overwrite_existing);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        pcb_vision::configure_utf8_console();
        const std::vector<std::string> utf8_args = pcb_vision::command_line_to_utf8_args(argc, argv);
        const Args args = parse_args(utf8_args);
        const fs::path output_dir = pcb_vision::path_from_utf8(args.output);
        fs::create_directories(output_dir);
        fs::create_directories(output_dir / "cpu" / "predictions");
        fs::create_directories(output_dir / "cuda" / "predictions");
        fs::create_directories(output_dir / "comparisons");

        std::vector<fs::path> images = bench::collect_images(pcb_vision::path_from_utf8(args.images), bench::parse_extensions(args.extensions), args.recursive);
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
        pcb_vision::CudaProviderConfig cuda_config;
        cuda_config.device_id = args.device_id;
        cuda_config.cudnn_conv_algo_search = args.cudnn_conv_algo_search;

        pcb_vision::OnnxDetector cpu_detector(args.model, class_names, args.imgsz, "CPUExecutionProvider", cuda_config);
        pcb_vision::OnnxDetector cuda_detector(args.model, class_names, args.imgsz, "CUDAExecutionProvider", cuda_config);
        const std::vector<std::string> available_providers = pcb_vision::available_execution_providers();

        cv::Mat warmup_image = pcb_vision::load_bgr_image(images.front());
        pcb_vision::PreprocessResult warmup_preprocess = pcb_vision::preprocess_image(warmup_image, args.imgsz);
        for (int i = 0; i < args.warmup; ++i) {
            (void)cpu_detector.infer_preprocessed(warmup_preprocess, warmup_image.size(), args.conf, args.iou);
            (void)cuda_detector.infer_preprocessed(warmup_preprocess, warmup_image.size(), args.conf, args.iou);
        }

        std::vector<ImageReport> reports;
        reports.reserve(images.size());
        for (std::size_t index = 0; index < images.size(); ++index) {
            const fs::path& image_path = images[index];
            const cv::Mat image = pcb_vision::load_bgr_image(image_path);
            const auto preprocess_start = std::chrono::steady_clock::now();
            pcb_vision::PreprocessResult preprocess = pcb_vision::preprocess_image(image, args.imgsz);
            const double preprocess_ms = elapsed_ms(preprocess_start);

            const bool cuda_first = args.provider_order == "cuda-first" || (args.provider_order == "alternate" && index % 2 == 1);
            RunTimings cpu_runs;
            RunTimings cuda_runs;
            if (cuda_first) {
                cuda_runs = run_repeated(cuda_detector, preprocess, image.size(), args, preprocess_ms);
                cpu_runs = run_repeated(cpu_detector, preprocess, image.size(), args, preprocess_ms);
            } else {
                cpu_runs = run_repeated(cpu_detector, preprocess, image.size(), args, preprocess_ms);
                cuda_runs = run_repeated(cuda_detector, preprocess, image.size(), args, preprocess_ms);
            }

            ImageReport report;
            report.image_path = image_path;
            report.width = image.cols;
            report.height = image.rows;
            report.cpu_detection_count = static_cast<int>(cpu_runs.baseline.detections.size());
            report.cuda_detection_count = static_cast<int>(cuda_runs.baseline.detections.size());
            report.comparison = bench::compare_detections(cpu_runs.baseline.detections, cuda_runs.baseline.detections, args.match_iou);
            report.cpu_session = bench::calculate_stats(cpu_runs.session_run_ms);
            report.cuda_session = bench::calculate_stats(cuda_runs.session_run_ms);
            report.cpu_total = bench::calculate_stats(cpu_runs.total_ms);
            report.cuda_total = bench::calculate_stats(cuda_runs.total_ms);
            report.cpu_session_runs = cpu_runs.session_run_ms;
            report.cuda_session_runs = cuda_runs.session_run_ms;
            report.cpu_total_runs = cpu_runs.total_ms;
            report.cuda_total_runs = cuda_runs.total_ms;
            report.cpu_internal_mismatches = cpu_runs.internal_mismatches;
            report.cuda_internal_mismatches = cuda_runs.internal_mismatches;
            report.status = bench::judge_status(
                report.cpu_detection_count,
                report.cuda_detection_count,
                report.comparison,
                args.strict_confidence_tolerance,
                args.practical_confidence_tolerance,
                args.bbox_tolerance
            );
            report.failure_reason = bench::failure_reason(
                report.cpu_detection_count,
                report.cuda_detection_count,
                report.comparison,
                args.strict_confidence_tolerance,
                args.practical_confidence_tolerance,
                args.bbox_tolerance
            );

            const std::string base = bench::sanitize_filename(image_path);
            write_detection_json(output_dir / "cpu" / "predictions" / (base + ".json"), cpu_runs.baseline);
            write_detection_json(output_dir / "cuda" / "predictions" / (base + ".json"), cuda_runs.baseline);
            write_comparison_json(output_dir / "comparisons" / (base + ".json"), report);
            if (report.status != "PASS") {
                copy_failure(image_path, output_dir, report.failure_reason);
            }
            reports.push_back(report);
            std::cout << "[" << index + 1 << "/" << images.size() << "] " << pcb_vision::path_to_utf8(image_path.filename())
                      << " " << report.status << " CPU " << report.cpu_session.mean
                      << " ms / CUDA " << report.cuda_session.mean << " ms\n";
        }

        write_image_results_csv(output_dir / "image_results.csv", args, reports);
        write_timing_csv(output_dir / "timing_runs.csv", reports);
        write_environment_json(output_dir / "environment.json", args);
        write_summary_json(output_dir / "summary.json", args, available_providers, reports);

        const int pass_count = static_cast<int>(std::count_if(reports.begin(), reports.end(), [](const ImageReport& item) { return item.status == "PASS"; }));
        const int numerical_warning_count = static_cast<int>(std::count_if(
            reports.begin(), reports.end(), [](const ImageReport& item) { return item.status == "NUMERICAL_WARNING"; }
        ));
        const int fail_count = static_cast<int>(std::count_if(reports.begin(), reports.end(), [](const ImageReport& item) { return item.status == "FAIL"; }));
        const int structural_mismatch_count = static_cast<int>(std::count_if(
            reports.begin(), reports.end(), [](const ImageReport& item) {
                return bench::has_structural_mismatch(
                    item.cpu_detection_count,
                    item.cuda_detection_count,
                    item.comparison
                );
            }
        ));
        const int cpu_internal = std::accumulate(reports.begin(), reports.end(), 0, [](int sum, const ImageReport& item) { return sum + item.cpu_internal_mismatches; });
        const int cuda_internal = std::accumulate(reports.begin(), reports.end(), 0, [](int sum, const ImageReport& item) { return sum + item.cuda_internal_mismatches; });
        const bench::TimingStats cpu_session = merged_stats(reports, false, false);
        const bench::TimingStats cuda_session = merged_stats(reports, true, false);
        const bench::TimingStats cpu_total = merged_stats(reports, false, true);
        const bench::TimingStats cuda_total = merged_stats(reports, true, true);
        const auto max_confidence = std::max_element(
            reports.begin(), reports.end(), [](const ImageReport& left, const ImageReport& right) {
                return left.comparison.max_confidence_diff < right.comparison.max_confidence_diff;
            }
        );
        const auto max_bbox = std::max_element(
            reports.begin(), reports.end(), [](const ImageReport& left, const ImageReport& right) {
                return left.comparison.max_bbox_diff < right.comparison.max_bbox_diff;
            }
        );
        const std::string final_status = fail_count > 0
            ? "FAIL"
            : (numerical_warning_count > 0 ? "NUMERICAL_WARNING" : "PASS");

        std::cout << "=== C++ ONNX CPU vs CUDA Batch Benchmark ===\n";
        std::cout << "Images: " << reports.size() << '\n';
        std::cout << "PASS: " << pass_count
                  << "\nNUMERICAL_WARNING: " << numerical_warning_count
                  << "\nFAIL: " << fail_count
                  << "\nStructural detection mismatches: " << structural_mismatch_count << '\n';
        std::cout << "Strict confidence tolerance: " << args.strict_confidence_tolerance
                  << "\nPractical confidence tolerance: " << args.practical_confidence_tolerance
                  << "\nBBox tolerance: " << args.bbox_tolerance
                  << "\nMatch IoU threshold: " << args.match_iou
                  << "\nCUDA device id: " << cuda_detector.cuda_config().device_id
                  << "\ncuDNN convolution algorithm search: "
                  << cuda_detector.cuda_config().cudnn_conv_algo_search
                  << "\nMax confidence diff: "
                  << (max_confidence == reports.end() ? 0.0 : max_confidence->comparison.max_confidence_diff)
                  << "\nMax bbox diff: "
                  << (max_bbox == reports.end() ? 0.0 : max_bbox->comparison.max_bbox_diff)
                  << "\n\n";
        std::cout << "CPU Session.Run:\nMean: " << cpu_session.mean << "\nMedian: " << cpu_session.median << "\nP95: " << cpu_session.p95 << "\n\n";
        std::cout << "CUDA Session.Run:\nMean: " << cuda_session.mean << "\nMedian: " << cuda_session.median << "\nP95: " << cuda_session.p95 << "\n\n";
        std::cout << "Speedup:\nSession.Run mean: " << bench::speedup(cpu_session.mean, cuda_session.mean)
                  << "x\nSession.Run median: " << bench::speedup(cpu_session.median, cuda_session.median)
                  << "x\nEnd-to-end mean: " << bench::speedup(cpu_total.mean, cuda_total.mean) << "x\n\n";
        std::cout << "Validation mismatches:\nCPU internal: " << cpu_internal
                  << "\nCUDA internal: " << cuda_internal
                  << "\nCPU vs CUDA: " << (numerical_warning_count + fail_count) << "\n\n";
        std::cout << "Final status: " << final_status << '\n';
        std::cout << "Output: " << pcb_vision::path_to_utf8(output_dir) << '\n';
        return (args.fail_on_mismatch && final_status != "PASS") ? 2 : 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << '\n';
        return 1;
    }
}
