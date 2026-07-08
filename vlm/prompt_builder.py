from __future__ import annotations

from model.yolo_result import YoloResult


class PromptBuilder:
    def build_defect_prompt(self, yolo_result: YoloResult) -> str:
        # TODO: Build a prompt that asks the VLM to explain only detected NG regions.
        return f"Explain {yolo_result.defect_count} detected PCB defect(s)."
