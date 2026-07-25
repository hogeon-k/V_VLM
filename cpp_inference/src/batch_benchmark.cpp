#include "batch_benchmark.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace pcb_vision::benchmark {
namespace {

std::string normalize_extension(std::string extension) {
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (!extension.empty() && extension.front() != '.') {
        extension.insert(extension.begin(), '.');
    }
    return extension;
}

std::array<double, 4> bbox_xyxy(const Detection& detection) {
    return {
        static_cast<double>(detection.box.x),
        static_cast<double>(detection.box.y),
        static_cast<double>(detection.box.x + detection.box.width),
        static_cast<double>(detection.box.y + detection.box.height)
    };
}

}  // namespace

std::vector<std::filesystem::path> collect_images(
    const std::filesystem::path& image_root,
    const std::vector<std::string>& extensions,
    bool recursive
) {
    if (!std::filesystem::exists(image_root) || !std::filesystem::is_directory(image_root)) {
        throw std::runtime_error("Image directory does not exist: " + image_root.string());
    }

    std::set<std::string> allowed;
    for (const std::string& extension : extensions) {
        allowed.insert(normalize_extension(extension));
    }
    std::vector<std::filesystem::path> images;

    const auto accept = [&allowed](const std::filesystem::directory_entry& entry) {
        if (!entry.is_regular_file()) {
            return false;
        }
        std::string extension = entry.path().extension().string();
        extension = normalize_extension(extension);
        return allowed.count(extension) > 0;
    };

    if (recursive) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(image_root)) {
            if (accept(entry)) {
                images.push_back(entry.path());
            }
        }
    } else {
        for (const auto& entry : std::filesystem::directory_iterator(image_root)) {
            if (accept(entry)) {
                images.push_back(entry.path());
            }
        }
    }
    std::sort(images.begin(), images.end());
    return images;
}

std::vector<std::string> parse_extensions(const std::string& value) {
    std::vector<std::string> extensions;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }), item.end());
        if (!item.empty()) {
            extensions.push_back(normalize_extension(item));
        }
    }
    if (extensions.empty()) {
        extensions = {".jpg", ".jpeg", ".png", ".bmp"};
    }
    return extensions;
}

double bbox_iou(const Detection& first, const Detection& second) {
    const auto a = bbox_xyxy(first);
    const auto b = bbox_xyxy(second);
    const double x1 = std::max(a[0], b[0]);
    const double y1 = std::max(a[1], b[1]);
    const double x2 = std::min(a[2], b[2]);
    const double y2 = std::min(a[3], b[3]);
    const double inter = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    const double area_a = std::max(0.0, a[2] - a[0]) * std::max(0.0, a[3] - a[1]);
    const double area_b = std::max(0.0, b[2] - b[0]) * std::max(0.0, b[3] - b[1]);
    const double denom = area_a + area_b - inter;
    return denom <= 0.0 ? 0.0 : inter / denom;
}

double max_bbox_coordinate_diff(const Detection& first, const Detection& second) {
    const auto a = bbox_xyxy(first);
    const auto b = bbox_xyxy(second);
    double max_diff = 0.0;
    for (std::size_t index = 0; index < a.size(); ++index) {
        max_diff = std::max(max_diff, std::fabs(a[index] - b[index]));
    }
    return max_diff;
}

MatchResult compare_detections(
    const std::vector<Detection>& cpu_detections,
    const std::vector<Detection>& cuda_detections,
    double match_iou
) {
    struct CandidatePair {
        double iou = 0.0;
        std::size_t cpu_index = 0;
        std::size_t cuda_index = 0;
    };

    std::vector<CandidatePair> pairs;
    for (std::size_t cpu_index = 0; cpu_index < cpu_detections.size(); ++cpu_index) {
        for (std::size_t cuda_index = 0; cuda_index < cuda_detections.size(); ++cuda_index) {
            if (cpu_detections[cpu_index].class_id != cuda_detections[cuda_index].class_id) {
                continue;
            }
            const double iou = bbox_iou(cpu_detections[cpu_index], cuda_detections[cuda_index]);
            if (iou >= match_iou) {
                pairs.push_back(CandidatePair{iou, cpu_index, cuda_index});
            }
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const CandidatePair& left, const CandidatePair& right) {
        if (left.iou != right.iou) {
            return left.iou > right.iou;
        }
        if (left.cpu_index != right.cpu_index) {
            return left.cpu_index < right.cpu_index;
        }
        return left.cuda_index < right.cuda_index;
    });

    std::set<std::size_t> used_cpu;
    std::set<std::size_t> used_cuda;
    std::vector<double> conf_diffs;
    std::vector<double> bbox_diffs;
    std::vector<double> matched_ious;
    for (const CandidatePair& pair : pairs) {
        if (used_cpu.count(pair.cpu_index) > 0 || used_cuda.count(pair.cuda_index) > 0) {
            continue;
        }
        used_cpu.insert(pair.cpu_index);
        used_cuda.insert(pair.cuda_index);
        const Detection& cpu = cpu_detections[pair.cpu_index];
        const Detection& cuda = cuda_detections[pair.cuda_index];
        conf_diffs.push_back(std::fabs(static_cast<double>(cpu.confidence) - static_cast<double>(cuda.confidence)));
        bbox_diffs.push_back(max_bbox_coordinate_diff(cpu, cuda));
        matched_ious.push_back(pair.iou);
    }

    MatchResult result;
    result.matched_count = static_cast<int>(matched_ious.size());
    result.cpu_only_count = static_cast<int>(cpu_detections.size() - used_cpu.size());
    result.cuda_only_count = static_cast<int>(cuda_detections.size() - used_cuda.size());
    result.class_mismatch_count = 0;
    if (!conf_diffs.empty()) {
        result.avg_confidence_diff = std::accumulate(conf_diffs.begin(), conf_diffs.end(), 0.0) / conf_diffs.size();
        result.max_confidence_diff = *std::max_element(conf_diffs.begin(), conf_diffs.end());
        result.avg_bbox_diff = std::accumulate(bbox_diffs.begin(), bbox_diffs.end(), 0.0) / bbox_diffs.size();
        result.max_bbox_diff = *std::max_element(bbox_diffs.begin(), bbox_diffs.end());
        result.avg_matched_iou = std::accumulate(matched_ious.begin(), matched_ious.end(), 0.0) / matched_ious.size();
        result.min_matched_iou = *std::min_element(matched_ious.begin(), matched_ious.end());
    }
    return result;
}

TimingStats calculate_stats(const std::vector<double>& values) {
    if (values.empty()) {
        return {};
    }
    TimingStats stats;
    stats.count = static_cast<int>(values.size());
    stats.first = values.front();
    stats.min = *std::min_element(values.begin(), values.end());
    stats.max = *std::max_element(values.begin(), values.end());
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t middle = sorted.size() / 2;
    stats.median = sorted.size() % 2 == 0 ? (sorted[middle - 1] + sorted[middle]) / 2.0 : sorted[middle];
    const std::size_t p95_index = std::min(
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(sorted.size()))) - 1,
        sorted.size() - 1
    );
    stats.p95 = sorted[p95_index];

    double squared_sum = 0.0;
    for (double value : values) {
        const double diff = value - stats.mean;
        squared_sum += diff * diff;
    }
    stats.stddev = std::sqrt(squared_sum / values.size());
    return stats;
}

double speedup(double cpu_ms, double cuda_ms) {
    return cuda_ms <= 0.0 ? 0.0 : cpu_ms / cuda_ms;
}

std::string judge_status(
    int cpu_count,
    int cuda_count,
    const MatchResult& comparison,
    double confidence_tolerance,
    double bbox_tolerance
) {
    if (cpu_count == cuda_count
        && comparison.cpu_only_count == 0
        && comparison.cuda_only_count == 0
        && comparison.class_mismatch_count == 0
        && comparison.max_confidence_diff <= confidence_tolerance
        && comparison.max_bbox_diff <= bbox_tolerance) {
        return "PASS";
    }
    if (cpu_count == cuda_count
        && comparison.cpu_only_count == 0
        && comparison.cuda_only_count == 0
        && comparison.class_mismatch_count == 0) {
        return "WARNING";
    }
    return "FAIL";
}

std::string failure_reason(
    int cpu_count,
    int cuda_count,
    const MatchResult& comparison,
    double confidence_tolerance,
    double bbox_tolerance
) {
    std::ostringstream reason;
    if (cpu_count != cuda_count) {
        reason << "detection_count_mismatch ";
    }
    if (comparison.cpu_only_count > 0) {
        reason << "cpu_only ";
    }
    if (comparison.cuda_only_count > 0) {
        reason << "cuda_only ";
    }
    if (comparison.class_mismatch_count > 0) {
        reason << "class_mismatch ";
    }
    if (comparison.max_confidence_diff > confidence_tolerance) {
        reason << "confidence_mismatch ";
    }
    if (comparison.max_bbox_diff > bbox_tolerance) {
        reason << "bbox_mismatch ";
    }
    std::string text = reason.str();
    if (text.empty()) {
        return "within tolerances";
    }
    if (!text.empty() && text.back() == ' ') {
        text.pop_back();
    }
    return text;
}

std::string sanitize_filename(const std::filesystem::path& image_path) {
    std::string name = image_path.stem().string();
    for (char& ch : name) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_' && ch != '-') {
            ch = '_';
        }
    }
    return name.empty() ? "image" : name;
}

}  // namespace pcb_vision::benchmark
