#include "postprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

namespace pcb_vision {
namespace {

struct Candidate {
    Detection detection;
    int candidate_index = 0;
};

cv::Rect2f xywh_to_xyxy(float x_center, float y_center, float width, float height) {
    return cv::Rect2f(
        x_center - width / 2.0F,
        y_center - height / 2.0F,
        width,
        height
    );
}

bool score_desc_index_asc(const Candidate& first, const Candidate& second) {
    if (first.detection.confidence == second.detection.confidence) {
        return first.candidate_index < second.candidate_index;
    }
    return first.detection.confidence > second.detection.confidence;
}

}  // namespace

std::vector<Detection> decode_yolo_output(
    const float* output_data,
    const std::vector<int64_t>& output_shape,
    const LetterboxResult& letterbox,
    const cv::Size& original_size,
    float confidence_threshold,
    float nms_iou_threshold,
    const std::vector<std::string>& class_names
) {
    if (output_data == nullptr) {
        throw std::invalid_argument("ONNX output data pointer is null.");
    }
    if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] < 5) {
        throw std::runtime_error("Expected ONNX output shape [1, 7, N].");
    }

    const int64_t channel_count = output_shape[1];
    const int64_t candidate_count = output_shape[2];
    const int64_t class_count = channel_count - 4;
    if (class_count != static_cast<int64_t>(class_names.size())) {
        throw std::runtime_error("Output class count does not match metadata class_names.");
    }

    std::vector<Candidate> candidates;
    for (int64_t candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
        int best_class = 0;
        float best_score = output_data[4 * candidate_count + candidate_index];
        for (int64_t class_offset = 1; class_offset < class_count; ++class_offset) {
            const float score = output_data[(4 + class_offset) * candidate_count + candidate_index];
            if (score > best_score) {
                best_score = score;
                best_class = static_cast<int>(class_offset);
            }
        }
        if (best_score < confidence_threshold) {
            continue;
        }

        const float x_center = output_data[candidate_index];
        const float y_center = output_data[candidate_count + candidate_index];
        const float width = output_data[2 * candidate_count + candidate_index];
        const float height = output_data[3 * candidate_count + candidate_index];
        cv::Rect2f box = xywh_to_xyxy(x_center, y_center, width, height);
        box = restore_box_to_original_image(box, letterbox, original_size);
        if (box.width <= 0.0F || box.height <= 0.0F) {
            continue;
        }

        Detection detection;
        detection.class_id = best_class;
        detection.class_name = class_names.at(static_cast<std::size_t>(best_class));
        detection.confidence = best_score;
        detection.box = box;
        candidates.push_back(Candidate{detection, static_cast<int>(candidate_index)});
    }

    std::map<int, std::vector<Candidate>> by_class;
    for (const Candidate& candidate : candidates) {
        by_class[candidate.detection.class_id].push_back(candidate);
    }

    std::vector<Candidate> kept;
    for (auto& [class_id, class_candidates] : by_class) {
        (void)class_id;
        std::stable_sort(class_candidates.begin(), class_candidates.end(), score_desc_index_asc);
        std::vector<bool> suppressed(class_candidates.size(), false);
        for (std::size_t i = 0; i < class_candidates.size(); ++i) {
            if (suppressed[i]) {
                continue;
            }
            kept.push_back(class_candidates[i]);
            for (std::size_t j = i + 1; j < class_candidates.size(); ++j) {
                if (suppressed[j]) {
                    continue;
                }
                if (bbox_iou(class_candidates[i].detection.box, class_candidates[j].detection.box) > nms_iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    std::stable_sort(kept.begin(), kept.end(), score_desc_index_asc);

    std::vector<Detection> detections;
    detections.reserve(kept.size());
    for (const Candidate& candidate : kept) {
        detections.push_back(candidate.detection);
    }
    return detections;
}

std::vector<Detection> apply_nms(
    const std::vector<Detection>& detections,
    float nms_iou_threshold
) {
    std::vector<Candidate> candidates;
    candidates.reserve(detections.size());
    for (std::size_t index = 0; index < detections.size(); ++index) {
        candidates.push_back(Candidate{detections[index], static_cast<int>(index)});
    }
    std::stable_sort(candidates.begin(), candidates.end(), score_desc_index_asc);

    std::vector<Detection> kept;
    std::vector<bool> suppressed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        kept.push_back(candidates[i].detection);
        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            if (!suppressed[j] && candidates[i].detection.class_id == candidates[j].detection.class_id
                && bbox_iou(candidates[i].detection.box, candidates[j].detection.box) > nms_iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}

float bbox_iou(const cv::Rect2f& first, const cv::Rect2f& second) {
    const float inter_x1 = std::max(first.x, second.x);
    const float inter_y1 = std::max(first.y, second.y);
    const float inter_x2 = std::min(first.x + first.width, second.x + second.width);
    const float inter_y2 = std::min(first.y + first.height, second.y + second.height);
    const float inter_area = std::max(0.0F, inter_x2 - inter_x1) * std::max(0.0F, inter_y2 - inter_y1);
    const float first_area = std::max(0.0F, first.width) * std::max(0.0F, first.height);
    const float second_area = std::max(0.0F, second.width) * std::max(0.0F, second.height);
    const float union_area = first_area + second_area - inter_area;
    return union_area <= 0.0F ? 0.0F : inter_area / union_area;
}

cv::Rect2f restore_box_to_original_image(
    const cv::Rect2f& letterboxed_box,
    const LetterboxResult& letterbox,
    const cv::Size& original_size
) {
    const float x1 = (letterboxed_box.x - letterbox.pad_x) / letterbox.scale;
    const float y1 = (letterboxed_box.y - letterbox.pad_y) / letterbox.scale;
    const float x2 = (letterboxed_box.x + letterboxed_box.width - letterbox.pad_x) / letterbox.scale;
    const float y2 = (letterboxed_box.y + letterboxed_box.height - letterbox.pad_y) / letterbox.scale;

    const float clipped_x1 = std::clamp(x1, 0.0F, static_cast<float>(original_size.width));
    const float clipped_y1 = std::clamp(y1, 0.0F, static_cast<float>(original_size.height));
    const float clipped_x2 = std::clamp(x2, 0.0F, static_cast<float>(original_size.width));
    const float clipped_y2 = std::clamp(y2, 0.0F, static_cast<float>(original_size.height));
    return cv::Rect2f(clipped_x1, clipped_y1, clipped_x2 - clipped_x1, clipped_y2 - clipped_y1);
}

}  // namespace pcb_vision
