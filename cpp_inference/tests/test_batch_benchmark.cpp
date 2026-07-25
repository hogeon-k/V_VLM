#include "batch_benchmark.hpp"

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

void test_status_stats_and_speedup() {
    const bench::TimingStats stats = bench::calculate_stats({1.0, 2.0, 3.0, 4.0});
    require(stats.first == 1.0, "first stat failed");
    require(stats.mean == 2.5, "mean stat failed");
    require(stats.median == 2.5, "median stat failed");
    require(stats.p95 == 4.0, "p95 stat failed");
    require(bench::speedup(100.0, 25.0) == 4.0, "speedup failed");

    bench::MatchResult comparison;
    comparison.max_confidence_diff = 0.0005;
    comparison.max_bbox_diff = 0.5;
    require(bench::judge_status(1, 1, comparison, 0.001, 1.0) == "PASS", "PASS status failed");
    comparison.max_confidence_diff = 0.01;
    require(bench::judge_status(1, 1, comparison, 0.001, 1.0) == "WARNING", "WARNING status failed");
    comparison.cpu_only_count = 1;
    require(bench::judge_status(1, 1, comparison, 0.001, 1.0) == "FAIL", "FAIL status failed");
    require(bench::failure_reason(1, 1, comparison, 0.001, 1.0).find("cpu_only") != std::string::npos, "failure classification failed");
}

}  // namespace

int main() {
    try {
        test_extension_parsing_and_collection();
        test_iou_class_matching_and_differences();
        test_status_stats_and_speedup();
        std::cout << "batch benchmark tests passed\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
}
