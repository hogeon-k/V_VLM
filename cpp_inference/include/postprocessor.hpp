#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "inference_result.hpp"
#include "image_preprocessor.hpp"

namespace pcb_vision {

std::vector<Detection> decode_yolo_output(
    const float* output_data,
    const std::vector<int64_t>& output_shape,
    const LetterboxResult& letterbox,
    const cv::Size& original_size,
    float confidence_threshold,
    float nms_iou_threshold,
    const std::vector<std::string>& class_names
);

std::vector<Detection> apply_nms(
    const std::vector<Detection>& detections,
    float nms_iou_threshold
);

float bbox_iou(const cv::Rect2f& first, const cv::Rect2f& second);

cv::Rect2f restore_box_to_original_image(
    const cv::Rect2f& letterboxed_box,
    const LetterboxResult& letterbox,
    const cv::Size& original_size
);

}  // namespace pcb_vision
