from __future__ import annotations

from model.defect_info import Detection
from model.yolo_result import YoloResult


LOW_CONFIDENCE_THRESHOLD = 0.70

CLASS_VISUAL_GUIDES = {
    "open_circuit": (
        "focus on broken or missing circuit pattern, discontinuity, connection state, "
        "and differences from nearby normal traces"
    ),
    "short": (
        "focus on abnormal connection, contact, or bridge shape between traces that "
        "should be separated"
    ),
    "missing_hole": (
        "focus on a blocked, absent, or positionally different hole compared with "
        "nearby normal holes"
    ),
}

DEFAULT_VISUAL_GUIDE = (
    "focus on visible differences from nearby normal patterns and avoid guessing "
    "features that are not visible in the image"
)


class PromptBuilder:
    """Build prompts that keep YOLO data authoritative and VLM output structured."""

    def build_defect_prompt(self, yolo_result: YoloResult) -> str:
        detections = "\n\n".join(
            self._format_detection(index, detection)
            for index, detection in enumerate(yolo_result.detections, start=1)
        )
        class_guides = "\n".join(
            f"- {class_name}: {guide}"
            for class_name, guide in sorted(CLASS_VISUAL_GUIDES.items())
        )
        final_judgment = "NG" if yolo_result.is_ng else "OK"
        defect_classes = ", ".join(
            dict.fromkeys(detection.class_name for detection in yolo_result.detections)
        )

        return f"""You are an assistant that explains PCB defect inspection results to an operator.

Input images:
- Image 1 is the full YOLO result image with bounding boxes.
- Image 2 is a crop montage of the detected regions.
- The crop montage order matches Detection 1, Detection 2, Detection 3, and so on.

Your role:
- Explain the YOLO detection results in a way an operator can understand.
- Add only visual observations that can be checked from the full result image or crop montage.
- Do not repeat the general definition of the defect class as the visual_feature.
- The visual_feature must describe the actual shape, gap, color/contrast, continuity, or pattern seen in that specific crop.
- If the crop does not clearly show a concrete feature, say that the crop image does not clearly confirm the feature.
- Describe each detection separately.
- Do not change the YOLO final judgment, class name, detection count, location, confidence, or bounding box.
- Do not remove YOLO detections or rename them to another defect class.
- If something is not clearly visible, write that it is difficult to confirm from the image alone.
- You are not the final judge. You are an explanation system that supports operator review.

Authoritative YOLO summary:
- Final judgment: {final_judgment}
- Defect classes: {defect_classes or "none"}
- Detection count: {yolo_result.defect_count}

Required rules:
1. Use the YOLO final OK/NG judgment exactly as provided.
2. Use the YOLO defect class, detection count, location, bounding box, and confidence exactly as provided.
3. Do not delete a YOLO detection or change it to another class.
4. Explain every Detection individually.
5. Write only features that are actually observable in the images.
6. If the feature is unclear, write "image-only confirmation is difficult".
7. For open_circuit, focus on broken/missing trace pattern, discontinuity, connection state, and contrast with nearby normal traces.
8. Mention uncertainty when reflection, drawing/overlay, blur, crop scale, or low visibility may affect confirmation.
9. If confidence is below {LOW_CONFIDENCE_THRESHOLD:.2f}, mark it as priority recheck.
10. For an NG judgment, write at least one operator check item.
11. Do not write "no check required", "confirmation unnecessary", or "no additional action" for NG.
12. The VLM is only a visual explanation assistant; YOLO remains the source of truth.

Class-specific visual guides:
{class_guides}
- unlisted classes: {DEFAULT_VISUAL_GUIDE}

YOLO detections:
{detections}

Prefer JSON if possible, with this schema:
{{
  "final_judgment": "{final_judgment}",
  "defect_classes": ["{defect_classes}"],
  "detection_count": {yolo_result.defect_count},
  "detections": [
    {{
      "detection_id": 1,
      "class_name": "",
      "location": "",
      "confidence": 0.0000,
      "bounding_box": [0, 0, 0, 0],
      "visual_feature": "actual visible feature in this crop, not the class definition",
      "uncertainty": "",
      "operator_check": "",
      "priority_recheck": false
    }}
  ],
  "overall_reason": "",
  "final_operator_check": ""
}}

If you cannot produce valid JSON, use this exact text structure:
1. Final judgment: {final_judgment}

2. Detection summary:
   - Defect class: {defect_classes or "none"}
   - Detection count: {yolo_result.defect_count}

3. Detection details:
   - Detection 1:
     - Location:
     - Confidence:
     - Bounding Box:
     - Observed visual feature:
     - Uncertainty:
     - Operator check:
     - Priority recheck:

4. Overall judgment reason:

5. Final operator check:
"""

    def _format_detection(self, index: int, detection: Detection) -> str:
        priority_recheck = detection.confidence < LOW_CONFIDENCE_THRESHOLD
        guide = CLASS_VISUAL_GUIDES.get(detection.class_name, DEFAULT_VISUAL_GUIDE)
        return (
            f"- Detection {index}:\n"
            f"  - Class: {detection.class_name}\n"
            f"  - Location: {detection.location or 'location unavailable'}\n"
            f"  - Confidence: {detection.confidence:.4f}\n"
            f"  - Bounding Box: ({detection.x1}, {detection.y1}, {detection.x2}, {detection.y2})\n"
            f"  - Visual guide: {guide}\n"
            f"  - Priority recheck: {'yes' if priority_recheck else 'no'}"
        )
