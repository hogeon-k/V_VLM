#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "inference_result.hpp"

namespace pcb_vision::benchmark {

struct TimingStats {
    int count = 0;
    double first = 0.0;
    double min = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

struct MatchResult {
    int matched_count = 0;
    int cpu_only_count = 0;
    int cuda_only_count = 0;
    int class_mismatch_count = 0;
    double avg_confidence_diff = 0.0;
    double max_confidence_diff = 0.0;
    double avg_bbox_diff = 0.0;
    double max_bbox_diff = 0.0;
    double avg_matched_iou = 0.0;
    double min_matched_iou = 1.0;
};

std::vector<std::filesystem::path> collect_images(
    const std::filesystem::path& image_root,
    const std::vector<std::string>& extensions,
    bool recursive
);

std::vector<std::string> parse_extensions(const std::string& value);

double bbox_iou(const Detection& first, const Detection& second);

double max_bbox_coordinate_diff(const Detection& first, const Detection& second);

MatchResult compare_detections(
    const std::vector<Detection>& cpu_detections,
    const std::vector<Detection>& cuda_detections,
    double match_iou
);

TimingStats calculate_stats(const std::vector<double>& values);

double speedup(double cpu_ms, double cuda_ms);

std::string judge_status(
    int cpu_count,
    int cuda_count,
    const MatchResult& comparison,
    double strict_confidence_tolerance,
    double practical_confidence_tolerance,
    double bbox_tolerance
);

std::string failure_reason(
    int cpu_count,
    int cuda_count,
    const MatchResult& comparison,
    double strict_confidence_tolerance,
    double practical_confidence_tolerance,
    double bbox_tolerance
);

bool has_structural_mismatch(
    int cpu_count,
    int cuda_count,
    const MatchResult& comparison
);

void validate_confidence_tolerances(
    double strict_confidence_tolerance,
    double practical_confidence_tolerance
);

std::string sanitize_filename(const std::filesystem::path& image_path);

}  // namespace pcb_vision::benchmark
