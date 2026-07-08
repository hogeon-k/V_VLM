from __future__ import annotations


class VlmClient:
    def generate(self, prompt: str, image_bytes: bytes | None = None) -> str:
        # TODO: Call the selected VLM provider after the provider is chosen.
        raise NotImplementedError
