from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import scripts.test_vlm_repeatability as repeatability
from model.defect_info import Detection
from model.yolo_result import YoloResult
from scripts.test_vlm_repeatability import (
    canonical_parsed_json,
    compare_values,
    exact_match_label,
    failure_signature,
    sha256_text,
)
from vlm.ollama_response import OllamaResponseMetadata
from vlm.response_parser import (
    ParsedVlmDetection,
    ParsedVlmResponse,
    VlmParseResult,
    VlmQualityInfo,
)


def test_canonical_parsed_json_is_sorted_and_compact() -> None:
    parse_result = VlmParseResult(
        raw_response="raw",
        parse_success=True,
        parse_error="",
        fallback_used=False,
        parsed_response=ParsedVlmResponse(
            final_judgment="NG",
            detections=[
                ParsedVlmDetection(
                    detection_id=1,
                    visual_feature="feature",
                    visibility="clear",
                    review_required=False,
                )
            ],
            summary="summary",
            raw_data={"summary": "summary", "final_judgment": "NG", "detections": []},
        ),
        formatted_response="formatted",
    )

    assert canonical_parsed_json(parse_result) == json.dumps(
        {"detections": [], "final_judgment": "NG", "summary": "summary"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_canonical_parsed_json_is_empty_on_fallback() -> None:
    assert canonical_parsed_json(None) == ""


def test_compare_values_counts_matches_against_first_value() -> None:
    assert compare_values(["a", "a", "b"]) == (2, False)
    assert compare_values(["a", "a"]) == (2, True)
    assert compare_values([]) == (0, True)


def test_sha256_text_is_stable() -> None:
    assert sha256_text("same") == sha256_text("same")
    assert sha256_text("same") != sha256_text("different")


def test_exact_match_label_returns_na_for_too_few_values() -> None:
    assert exact_match_label([]) == "N/A"
    assert exact_match_label(["only"]) == "N/A"


def test_exact_match_label_compares_two_or_more_values() -> None:
    assert exact_match_label(["a", "a", "a"]) == "true"
    assert exact_match_label(["a", "b", "a"]) == "false"


def test_failure_signature_uses_stable_status_fields() -> None:
    class FakeService:
        last_vlm_status = "done_false"
        last_parse_status = "not_attempted"
        last_fallback_used = True
        last_ollama_metadata = OllamaResponseMetadata(done=False, content_length=0)

    assert failure_signature(FakeService()) == (
        '{"fallback_used":true,"ollama_content_length":0,'
        '"ollama_done":false,"parse_status":"not_attempted","vlm_status":"done_false"}'
    )


def test_repeatability_cli_accepts_vlm_size_experiment_options(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_vlm_repeatability.py",
            "--image",
            "sample.jpg",
            "--vlm-full-image-size",
            "896",
            "--vlm-montage-size",
            "960",
            "--vlm-image-mode",
            "full",
        ],
    )

    args = repeatability.parse_args()

    assert args.vlm_full_image_size == 896
    assert args.vlm_montage_size == 960
    assert args.vlm_image_mode == "full"


def test_repeatability_main_prints_quality_aggregates(monkeypatch, capsys) -> None:
    yolo_result = YoloResult(
        image_path=Path("sample.jpg"),
        detections=[Detection(0, "short", 0.9, 1, 2, 3, 4)],
    )

    class FakeYoloService:
        def detect(self, image_path: Path) -> YoloResult:
            return yolo_result

    qualities = [
        VlmQualityInfo(quality_status="warning", class_name_only_count=1, semantic_warning_count=1),
        VlmQualityInfo(quality_status="acceptable"),
    ]

    class FakeVlmService:
        def __init__(self, quality: VlmQualityInfo) -> None:
            self.last_quality_info = quality
            self.last_raw_response = json.dumps({"run": quality.quality_status})
            self.last_preparation_info = SimpleNamespace(
                image_mode="full_montage",
                image_count=2,
                full_image_size=(640, 335),
                crop_montage_size=(640, 640),
            )
            self.last_parse_result = VlmParseResult(
                raw_response=self.last_raw_response,
                parse_success=True,
                parse_error="",
                fallback_used=False,
                parsed_response=ParsedVlmResponse(
                    final_judgment="NG",
                    detections=[
                        ParsedVlmDetection(
                            detection_id=1,
                            visual_feature="feature",
                            visibility="clear",
                            review_required=False,
                        )
                    ],
                    summary="summary",
                    raw_data={"final_judgment": "NG", "detections": [], "summary": "summary"},
                ),
                formatted_response="formatted",
                quality_info=quality,
            )
            self.last_vlm_status = "success"
            self.last_parse_status = "success"
            self.last_fallback_used = False
            self.last_ollama_metadata = None

        def describe_defects(self, image_path: Path, yolo_result: YoloResult) -> str:
            return "formatted"

    services = [FakeVlmService(quality) for quality in qualities]
    monkeypatch.setattr(repeatability, "build_yolo_service", lambda args: FakeYoloService())
    monkeypatch.setattr(repeatability, "build_vlm_service", lambda args: services.pop(0))
    monkeypatch.setattr(
        sys,
        "argv",
        ["test_vlm_repeatability.py", "--image", "sample.jpg", "--repeat-count", "2"],
    )

    assert repeatability.main() == 0

    output = capsys.readouterr().out
    assert "Quality acceptable count: 1" in output
    assert "Quality warning count: 1" in output
    assert "Total class-name-only count: 1" in output
    assert "Class-name-only run rate: 50.0%" in output
