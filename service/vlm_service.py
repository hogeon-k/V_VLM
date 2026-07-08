from __future__ import annotations

from pathlib import Path

from model.yolo_result import YoloResult
from vlm.prompt_builder import PromptBuilder
from vlm.response_parser import VlmResponseParser
from vlm.vlm_client import VlmClient


class VlmService:
    def __init__(
        self,
        client: VlmClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: VlmResponseParser | None = None,
    ) -> None:
        self.client = client or VlmClient()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_parser = response_parser or VlmResponseParser()

    def describe_defects(self, image_path: Path, yolo_result: YoloResult) -> str | None:
        # TODO: Build a provider-neutral prompt and parse the VLM response.
        return None
