#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "inference_result.hpp"
#include "image_preprocessor.hpp"

namespace pcb_vision {

std::vector<Detection> decode_yolo_output_placeholder(
    const cv::Mat& output,
    float confidence_threshold,
    const std::vector<std::string>& class_names
);

std::vector<Detection> apply_nms(
    const std::vector<Detection>& detections,
    float nms_iou_threshold
);

cv::Rect restore_box_to_original_image(
    const cv::Rect& letterboxed_box,
    const LetterboxResult& letterbox,
    const cv::Size& original_size
);

}  // namespace pcb_vision
