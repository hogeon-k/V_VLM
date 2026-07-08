from __future__ import annotations


class VlmResponseParser:
    def parse_description(self, response_text: str) -> str:
        # TODO: Normalize provider-specific VLM responses.
        return response_text.strip()
