from __future__ import annotations

from model.defect_info import Detection
from model.yolo_result import YoloResult
from vlm.prompt_builder import CLASS_VISUAL_GUIDES, PromptBuilder


def test_prompt_builder_includes_detection_metadata_and_rules() -> None:
    result = YoloResult(
        image_path="sample.jpg",
        detections=[
            Detection(0, "open_circuit", 0.784, 2711, 946, 2739, 979, location="middle right"),
            Detection(0, "open_circuit", 0.6696, 2193, 294, 2230, 326, location="upper right"),
        ],
    )

    prompt = PromptBuilder().build_defect_prompt(result)

    assert "Final judgment: NG" in prompt
    assert "Detection count: 2" in prompt
    assert "- Detection 1:" in prompt
    assert "- Class: open_circuit" in prompt
    assert "- Location: middle right" in prompt
    assert "- Confidence: 0.7840" in prompt
    assert "- Bounding Box: (2711, 946, 2739, 979)" in prompt
    assert "- Priority recheck: yes" in prompt
    assert "Do not change the YOLO final judgment" in prompt
    assert "Do not delete a YOLO detection" in prompt
    assert "Do not write \"no check required\"" in prompt
    assert CLASS_VISUAL_GUIDES["open_circuit"] in prompt


def test_prompt_builder_marks_low_confidence_priority_recheck() -> None:
    result = YoloResult(
        image_path="sample.jpg",
        detections=[Detection(0, "open_circuit", 0.6696, 1, 2, 3, 4)],
    )

    prompt = PromptBuilder().build_defect_prompt(result)

    assert "Confidence: 0.6696" in prompt
    assert "Priority recheck: yes" in prompt
    assert "If confidence is below 0.70" in prompt
