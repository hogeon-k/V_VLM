#include "postprocessor.hpp"

#include <opencv2/dnn.hpp>

#include <algorithm>

namespace pcb_vision {

std::vector<Detection> decode_yolo_output_placeholder(
    const cv::Mat& output,
    float confidence_threshold,
    const std::vector<std::string>& class_names
) {
    (void)output;
    (void)confidence_threshold;
    (void)class_names;
    return {};
}

std::vector<Detection> apply_nms(
    const std::vector<Detection>& detections,
    float nms_iou_threshold
) {
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    boxes.reserve(detections.size());
    scores.reserve(detections.size());

    for (const Detection& detection : detections) {
        boxes.push_back(detection.box);
        scores.push_back(detection.confidence);
    }

    std::vector<int> kept_indices;
    cv::dnn::NMSBoxes(boxes, scores, 0.0F, nms_iou_threshold, kept_indices);

    std::vector<Detection> kept;
    kept.reserve(kept_indices.size());
    for (int index : kept_indices) {
        kept.push_back(detections.at(static_cast<std::size_t>(index)));
    }
    return kept;
}

cv::Rect restore_box_to_original_image(
    const cv::Rect& letterboxed_box,
    const LetterboxResult& letterbox,
    const cv::Size& original_size
) {
    const int x = static_cast<int>((letterboxed_box.x - letterbox.pad_x) / letterbox.scale);
    const int y = static_cast<int>((letterboxed_box.y - letterbox.pad_y) / letterbox.scale);
    const int width = static_cast<int>(letterboxed_box.width / letterbox.scale);
    const int height = static_cast<int>(letterboxed_box.height / letterbox.scale);

    cv::Rect restored(x, y, width, height);
    const cv::Rect image_bounds(0, 0, original_size.width, original_size.height);
    return restored & image_bounds;
}

}  // namespace pcb_vision
