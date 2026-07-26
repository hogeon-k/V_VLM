#include "batch_benchmark.hpp"
#include "detector.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace bench = pcb_vision::benchmark;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

pcb_vision::Detection detection(int class_id, float confidence, float x1, float y1, float x2, float y2) {
    pcb_vision::Detection det;
    det.class_id = class_id;
    det.class_name = std::to_string(class_id);
    det.confidence = confidence;
    det.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
    return det;
}

void test_extension_parsing_and_collection() {
    const fs::path root = fs::temp_directory_path() / "pcb_batch_collect_test";
    fs::remove_all(root);
    fs::create_directories(root / "nested");
    std::ofstream(root / "a.jpg").put('x');
    std::ofstream(root / "b.PNG").put('x');
    std::ofstream(root / "c.txt").put('x');
    std::ofstream(root / "nested" / "d.bmp").put('x');

    const auto extensions = bench::parse_extensions("jpg,png");
    require(extensions.size() == 2 && extensions[0] == ".jpg" && extensions[1] == ".png", "extension parsing failed");
    require(bench::collect_images(root, extensions, false).size() == 2, "non-recursive image collection failed");
    require(bench::collect_images(root, bench::parse_extensions("jpg,png,bmp"), true).size() == 3, "recursive image collection failed");
    fs::remove_all(root);
}

void test_iou_class_matching_and_differences() {
    const std::vector<pcb_vision::Detection> cpu = {
        detection(1, 0.900F, 0.0F, 0.0F, 10.0F, 10.0F),
        detection(2, 0.800F, 20.0F, 20.0F, 40.0F, 40.0F)
    };
    const std::vector<pcb_vision::Detection> cuda = {
        detection(1, 0.899F, 0.5F, 0.0F, 10.5F, 10.0F),
        detection(2, 0.805F, 21.0F, 20.0F, 41.0F, 40.0F)
    };

    const bench::MatchResult result = bench::compare_detections(cpu, cuda, 0.5);
    require(result.matched_count == 2, "class-aware matching failed");
    require(result.cpu_only_count == 0 && result.cuda_only_count == 0, "unexpected unmatched detections");
    require(std::fabs(result.max_confidence_diff - 0.005F) < 0.0001, "confidence diff failed");
    require(std::fabs(result.max_bbox_diff - 1.0) < 0.0001, "bbox diff failed");
}

void require_status(
    double confidence_diff,
    double bbox_diff,
    const std::string& expected,
    const std::string& message
) {
    bench::MatchResult comparison;
    comparison.max_confidence_diff = confidence_diff;
    comparison.max_bbox_diff = bbox_diff;
    require(
        bench::judge_status(1, 1, comparison, 0.001, 0.002, 1.0) == expected,
        message
    );
}

void test_status_boundaries() {
    require_status(0.0009, 0.5, "PASS", "0.0009 confidence diff should pass");
    require_status(0.0010, 0.5, "PASS", "strict confidence boundary should pass");
    require_status(0.0015, 0.5, "NUMERICAL_WARNING", "confidence diff between tolerances should warn");
    require_status(0.0020, 0.5, "NUMERICAL_WARNING", "practical confidence boundary should warn");
    require_status(0.0021, 0.5, "FAIL", "confidence diff above practical tolerance should fail");
    require_status(0.0005, 1.0001, "FAIL", "bbox diff above tolerance should fail");

    bench::MatchResult comparison;
    comparison.class_mismatch_count = 1;
    require(
        bench::judge_status(1, 1, comparison, 0.001, 0.002, 1.0) == "FAIL",
        "class mismatch should fail"
    );

    comparison = {};
    comparison.cpu_only_count = 1;
    require(
        bench::judge_status(1, 1, comparison, 0.001, 0.002, 1.0) == "FAIL",
        "CPU-only detection should fail"
    );

    comparison = {};
    comparison.cuda_only_count = 1;
    require(
        bench::judge_status(1, 1, comparison, 0.001, 0.002, 1.0) == "FAIL",
        "CUDA-only detection should fail"
    );

    comparison = {};
    require(
        bench::judge_status(1, 2, comparison, 0.001, 0.002, 1.0) == "FAIL",
        "detection count mismatch should fail"
    );
}

void test_tolerance_validation_and_failure_reason() {
    bench::validate_confidence_tolerances(0.001, 0.002);

    bool negative_strict_rejected = false;
    try {
        bench::validate_confidence_tolerances(-0.001, 0.002);
    } catch (const std::invalid_argument& exc) {
        negative_strict_rejected = std::string(exc.what()).find(
            "--strict-confidence-tolerance"
        ) != std::string::npos;
    }
    require(negative_strict_rejected, "negative strict tolerance should be rejected");

    bool negative_practical_rejected = false;
    try {
        bench::validate_confidence_tolerances(0.001, -0.002);
    } catch (const std::invalid_argument& exc) {
        negative_practical_rejected = std::string(exc.what()).find(
            "--practical-confidence-tolerance"
        ) != std::string::npos;
    }
    require(negative_practical_rejected, "negative practical tolerance should be rejected");

    bool strict_greater_rejected = false;
    try {
        bench::validate_confidence_tolerances(0.0021, 0.002);
    } catch (const std::invalid_argument& exc) {
        strict_greater_rejected = std::string(exc.what()).find(
            "must be <= --practical-confidence-tolerance"
        ) != std::string::npos;
    }
    require(strict_greater_rejected, "strict > practical tolerance should be rejected");

    bench::MatchResult comparison;
    comparison.max_confidence_diff = 0.0015;
    require(
        bench::failure_reason(1, 1, comparison, 0.001, 0.002, 1.0) == "numerical_difference",
        "numerical warning reason failed"
    );
    comparison.max_confidence_diff = 0.0021;
    require(
        bench::failure_reason(1, 1, comparison, 0.001, 0.002, 1.0) == "confidence_mismatch",
        "practical tolerance failure reason failed"
    );
}

void test_cuda_algorithm_configuration() {
    const pcb_vision::CudaProviderConfig default_config;
    require(
        default_config.cudnn_conv_algo_search == "HEURISTIC",
        "default CUDA algorithm search should be HEURISTIC"
    );
    require(
        pcb_vision::normalize_cudnn_conv_algo_search("heuristic") == "HEURISTIC",
        "lowercase heuristic normalization failed"
    );
    require(
        pcb_vision::normalize_cudnn_conv_algo_search("HEURISTIC") == "HEURISTIC",
        "uppercase heuristic normalization failed"
    );
    require(
        pcb_vision::normalize_cudnn_conv_algo_search("Heuristic") == "HEURISTIC",
        "mixed-case heuristic normalization failed"
    );
    require(
        pcb_vision::normalize_cudnn_conv_algo_search("exhaustive") == "EXHAUSTIVE",
        "lowercase exhaustive normalization failed"
    );
    require(
        pcb_vision::normalize_cudnn_conv_algo_search("EXHAUSTIVE") == "EXHAUSTIVE",
        "uppercase exhaustive normalization failed"
    );
    require(
        pcb_vision::normalize_cudnn_conv_algo_search("Exhaustive") == "EXHAUSTIVE",
        "mixed-case exhaustive normalization failed"
    );
    require(
        pcb_vision::normalize_cudnn_conv_algo_search("default") == "DEFAULT",
        "lowercase default normalization failed"
    );
    require(
        pcb_vision::normalize_cudnn_conv_algo_search("DEFAULT") == "DEFAULT",
        "uppercase default normalization failed"
    );
    require(
        pcb_vision::normalize_cudnn_conv_algo_search("Default") == "DEFAULT",
        "mixed-case default normalization failed"
    );

    const auto require_invalid = [](const std::string& value) {
        bool rejected = false;
        try {
            (void)pcb_vision::normalize_cudnn_conv_algo_search(value);
        } catch (const std::invalid_argument& exc) {
            const std::string message = exc.what();
            rejected = message.find("Invalid cudnn_conv_algo_search value: " + value) != std::string::npos
                && message.find("Allowed values: heuristic, exhaustive, default") != std::string::npos;
        }
        require(rejected, "invalid CUDA algorithm search should be rejected: " + value);
    };
    require_invalid("invalid");
    require_invalid("fast");
    require_invalid("");
}

void test_stats_and_speedup() {
    const bench::TimingStats stats = bench::calculate_stats({1.0, 2.0, 3.0, 4.0});
    require(stats.first == 1.0, "first stat failed");
    require(stats.mean == 2.5, "mean stat failed");
    require(stats.median == 2.5, "median stat failed");
    require(stats.p95 == 4.0, "p95 stat failed");
    require(bench::speedup(100.0, 25.0) == 4.0, "speedup failed");
}

}  // namespace

int main() {
    try {
        test_extension_parsing_and_collection();
        test_iou_class_matching_and_differences();
        test_status_boundaries();
        test_tolerance_validation_and_failure_reason();
        test_cuda_algorithm_configuration();
        test_stats_and_speedup();
        std::cout << "batch benchmark tests passed\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
}
