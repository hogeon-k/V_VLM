from __future__ import annotations

import csv
import sys
from pathlib import Path

from model.defect_info import Detection
from model.yolo_result import YoloResult
import scripts.run_vlm_test_batch as batch
from scripts.run_vlm_test_batch import (
    CSV_COLUMNS,
    SKIPPED_VLM_MESSAGE,
    discover_images,
    ground_truth_for_category,
    result_to_row,
    write_csv,
)
from vlm.ollama_response import OllamaResponseMetadata
from vlm.response_parser import VlmQualityInfo


def test_discover_images_recurses_supported_extensions_only(tmp_path) -> None:
    (tmp_path / "open_circuit").mkdir()
    (tmp_path / "open_circuit" / "a.JPG").write_bytes(b"image")
    (tmp_path / "open_circuit" / "b.webp").write_bytes(b"image")
    (tmp_path / "open_circuit" / "README.md").write_text("ignore", encoding="utf-8")
    (tmp_path / "open_circuit" / ".gitkeep").write_text("", encoding="utf-8")
    (tmp_path / "normal").mkdir()
    (tmp_path / "normal" / "c.png").write_bytes(b"image")

    images = discover_images(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in images] == [
        "normal/c.png",
        "open_circuit/a.JPG",
        "open_circuit/b.webp",
    ]


def test_ground_truth_for_category_only_uses_definitive_classes() -> None:
    assert ground_truth_for_category("open_circuit") == "open_circuit"
    assert ground_truth_for_category("normal") == "normal"
    assert ground_truth_for_category("low_confidence") == ""
    assert ground_truth_for_category("false_positive_candidates") == ""


def test_result_to_row_flattens_multiple_detections(tmp_path) -> None:
    image_path = tmp_path / "open_circuit" / "open_circuit_001.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"image")
    yolo_result = YoloResult(
        image_path=image_path,
        detections=[
            Detection(0, "open_circuit", 0.784, 1, 2, 3, 4),
            Detection(1, "short", 0.6696, 5, 6, 7, 8),
        ],
        annotated_image_path=tmp_path / "result.jpg",
    )

    row = result_to_row(
        image_path=image_path,
        input_dir=tmp_path,
        yolo_result=yolo_result,
        vlm_model="qwen2.5vl:3b",
        vlm_response="response",
        vlm_raw_response='{"raw": true}',
        vlm_parse_success=True,
        vlm_parse_error="",
        vlm_fallback_used=False,
        vlm_temperature=0.0,
        vlm_top_p=0.8,
        vlm_top_k=20,
        vlm_repeat_penalty=1.1,
        vlm_seed=42,
        vlm_image_mode="full",
        crop_montage_path=tmp_path / "montage.jpg",
        full_image_size_limit=640,
        montage_size_limit=768,
        full_image_size=(640, 335),
        montage_size=(640, 640),
        image_preparation_seconds=1.2345,
        vlm_inference_seconds=2.3456,
        total_processing_seconds=3.4567,
        status="success",
        quality_status="warning",
        class_name_only_count=2,
        summary_contradiction=True,
        semantic_warning_count=3,
        class_name_only_detection_ids=(1, 2),
    )

    assert row["category"] == "open_circuit"
    assert row["ground_truth_class"] == "open_circuit"
    assert row["yolo_judgment"] == "NG"
    assert row["yolo_detection_count"] == 2
    assert row["yolo_classes"] == "open_circuit|short"
    assert row["yolo_confidences"] == "0.7840|0.6696"
    assert row["vlm_raw_response"] == '{"raw": true}'
    assert row["vlm_parse_success"] == "true"
    assert row["vlm_fallback_used"] == "false"
    assert row["vlm_temperature"] == 0.0
    assert row["vlm_top_p"] == 0.8
    assert row["vlm_top_k"] == 20
    assert row["vlm_repeat_penalty"] == 1.1
    assert row["vlm_seed"] == 42
    assert row["vlm_image_mode"] == "full"
    assert row["vlm_full_image_size_limit"] == 640
    assert row["vlm_montage_size_limit"] == 768
    assert row["vlm_full_image_width"] == 640
    assert row["vlm_full_image_height"] == 335
    assert row["montage_width"] == 640
    assert row["montage_height"] == 640
    assert row["quality_status"] == "warning"
    assert row["class_name_only_count"] == 2
    assert row["summary_contradiction"] == "true"
    assert row["semantic_warning_count"] == 3
    assert row["class_name_only_detection_ids"] == "1|2"
    assert row["image_preparation_time_seconds"] == "1.234"


def test_result_to_row_records_normal_no_detection_skip(tmp_path) -> None:
    image_path = tmp_path / "normal" / "normal_001.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"image")
    yolo_result = YoloResult(image_path=image_path, detections=[])

    row = result_to_row(
        image_path=image_path,
        input_dir=tmp_path,
        yolo_result=yolo_result,
        vlm_model="qwen2.5vl:3b",
        vlm_response=SKIPPED_VLM_MESSAGE,
        crop_montage_path=None,
        image_preparation_seconds=None,
        vlm_inference_seconds=None,
        total_processing_seconds=0.1,
        status="success",
    )

    assert row["ground_truth_class"] == "normal"
    assert row["yolo_judgment"] == "OK"
    assert row["yolo_detection_count"] == 0
    assert row["crop_montage_path"] == ""
    assert row["vlm_response"] == SKIPPED_VLM_MESSAGE
    assert row["vlm_parse_success"] == "false"
    assert row["vlm_fallback_used"] == "false"


def test_write_csv_uses_excel_friendly_utf8_bom(tmp_path) -> None:
    csv_path = tmp_path / "results.csv"
    row = {column: "" for column in CSV_COLUMNS}
    row["status"] = "success"

    write_csv(csv_path, [row])

    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["status"] == "success"


def test_csv_columns_include_parse_metadata_and_generation_options() -> None:
    assert "vlm_raw_response" in CSV_COLUMNS
    assert "vlm_parse_success" in CSV_COLUMNS
    assert "vlm_parse_error" in CSV_COLUMNS
    assert "vlm_fallback_used" in CSV_COLUMNS
    assert "vlm_temperature" in CSV_COLUMNS
    assert "vlm_top_p" in CSV_COLUMNS
    assert "vlm_top_k" in CSV_COLUMNS
    assert "vlm_repeat_penalty" in CSV_COLUMNS
    assert "vlm_seed" in CSV_COLUMNS
    assert "vlm_image_mode" in CSV_COLUMNS
    assert "pipeline_status" in CSV_COLUMNS
    assert "yolo_status" in CSV_COLUMNS
    assert "vlm_status" in CSV_COLUMNS
    assert "parse_status" in CSV_COLUMNS
    assert "fallback_used" in CSV_COLUMNS
    assert "ollama_done" in CSV_COLUMNS
    assert "ollama_content_length" in CSV_COLUMNS
    assert "vlm_full_image_size_limit" in CSV_COLUMNS
    assert "vlm_montage_size_limit" in CSV_COLUMNS
    assert "vlm_full_image_width" in CSV_COLUMNS
    assert "vlm_full_image_height" in CSV_COLUMNS
    assert "quality_status" in CSV_COLUMNS
    assert "class_name_only_count" in CSV_COLUMNS
    assert "summary_contradiction" in CSV_COLUMNS
    assert "semantic_warning_count" in CSV_COLUMNS
    assert "class_name_only_detection_ids" in CSV_COLUMNS


def test_write_csv_preserves_multiline_values(tmp_path) -> None:
    csv_path = tmp_path / "results.csv"
    row = {column: "" for column in CSV_COLUMNS}
    row["vlm_raw_response"] = "{\n  \"summary\": \"line\"\n}"
    row["vlm_response"] = "line 1\nline 2"
    row["status"] = "success"

    write_csv(csv_path, [row])

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["vlm_raw_response"] == "{\n  \"summary\": \"line\"\n}"
    assert rows[0]["vlm_response"] == "line 1\nline 2"


def test_result_to_row_records_done_false_empty_content_metadata(tmp_path) -> None:
    image_path = tmp_path / "short" / "short_001.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"image")
    yolo_result = YoloResult(
        image_path=image_path,
        detections=[Detection(1, "short", 0.9, 1, 2, 3, 4)],
    )

    row = result_to_row(
        image_path=image_path,
        input_dir=tmp_path,
        yolo_result=yolo_result,
        vlm_model="qwen2.5vl:3b",
        vlm_response="fallback",
        vlm_fallback_used=True,
        status="success",
        pipeline_status="success",
        yolo_status="success",
        vlm_status="done_false",
        parse_status="not_attempted",
        ollama_metadata=OllamaResponseMetadata(
            http_status=200,
            done=False,
            content_length=0,
            prompt_eval_count=0,
            eval_count=0,
            total_duration=0,
            load_duration=11,
            prompt_eval_duration=0,
            eval_duration=0,
        ),
        vlm_image_count=2,
        vlm_image_mode="full_montage",
        crop_count=1,
        full_image_size_limit=640,
        montage_size_limit=640,
        full_image_size=(640, 335),
        montage_size=(320, 240),
    )

    assert row["pipeline_status"] == "success"
    assert row["yolo_status"] == "success"
    assert row["vlm_status"] == "done_false"
    assert row["parse_status"] == "not_attempted"
    assert row["fallback_used"] == "true"
    assert row["ollama_done"] == "false"
    assert row["ollama_content_length"] == 0
    assert row["ollama_prompt_eval_count"] == 0
    assert row["ollama_eval_count"] == 0
    assert row["ollama_total_duration"] == 0
    assert row["ollama_load_duration"] == 11
    assert row["ollama_prompt_eval_duration"] == 0
    assert row["ollama_eval_duration"] == 0
    assert row["vlm_image_count"] == 2
    assert row["vlm_image_mode"] == "full_montage"
    assert row["crop_count"] == 1
    assert row["vlm_full_image_size_limit"] == 640
    assert row["vlm_montage_size_limit"] == 640
    assert row["vlm_full_image_width"] == 640
    assert row["vlm_full_image_height"] == 335
    assert row["montage_width"] == 320
    assert row["montage_height"] == 240


def test_batch_cli_accepts_vlm_size_experiment_options(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_vlm_test_batch.py",
            "--vlm-full-image-size",
            "640",
            "--vlm-montage-size",
            "768",
            "--vlm-image-mode",
            "montage",
        ],
    )

    args = batch.parse_args()

    assert args.vlm_full_image_size == 640
    assert args.vlm_montage_size == 768
    assert args.vlm_image_mode == "montage"


def test_result_to_row_records_yolo_failure_status(tmp_path) -> None:
    image_path = tmp_path / "short" / "short_001.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"image")

    row = result_to_row(
        image_path=image_path,
        input_dir=tmp_path,
        yolo_result=None,
        vlm_model="qwen2.5vl:3b",
        vlm_response="",
        status="error",
        pipeline_status="failed",
        yolo_status="failed",
        vlm_status="not_run",
        parse_status="not_attempted",
        exception_type="RuntimeError",
        exception_message="boom",
    )

    assert row["pipeline_status"] == "failed"
    assert row["yolo_status"] == "failed"
    assert row["vlm_status"] == "not_run"
    assert row["parse_status"] == "not_attempted"
    assert row["fallback_used"] == "false"
    assert row["exception_type"] == "RuntimeError"
    assert row["exception_message"] == "boom"


def test_batch_main_processes_and_saves_each_image_sequentially(monkeypatch, tmp_path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "batch"
    input_dir.mkdir()
    first = input_dir / "a.jpg"
    second = input_dir / "b.jpg"
    first.write_bytes(b"image")
    second.write_bytes(b"image")
    events: list[str] = []

    class FakeYoloService:
        def detect(self, image_path: Path, output_path: Path | None = None) -> YoloResult:
            if image_path == second:
                assert "save:a.jpg" in events
            events.append(f"yolo:{image_path.name}")
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"result")
            return YoloResult(
                image_path=image_path,
                detections=[Detection(0, "open_circuit", 0.9, 1, 2, 3, 4)],
                annotated_image_path=output_path,
            )

    class FakeVlmService:
        last_preparation_info = None
        last_raw_response = '{"ok": true}'
        last_parse_success = True
        last_parse_error = ""
        last_fallback_used = False
        last_vlm_status = "success"
        last_parse_status = "success"
        last_ollama_metadata = None
        last_quality_info = VlmQualityInfo(quality_status="acceptable")
        last_retry_count = 0
        last_failure_reason = ""

        def describe_defects(self, image_path: Path, yolo_result: YoloResult) -> str:
            events.append(f"vlm:{yolo_result.image_path.name}")
            return "description"

    original_write = batch.write_image_result_json

    def tracking_write(path: Path, row: dict[str, object], yolo_result: YoloResult | None) -> None:
        events.append(f"save:{row['image_name']}")
        original_write(path, row, yolo_result)

    monkeypatch.setattr(batch, "build_yolo_service", lambda args: FakeYoloService())
    monkeypatch.setattr(batch, "build_vlm_service", lambda args: FakeVlmService())
    monkeypatch.setattr(batch, "write_image_result_json", tracking_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_vlm_test_batch.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--vlm-retry-delay",
            "0",
        ],
    )

    assert batch.main() == 0

    assert events == [
        "yolo:a.jpg",
        "vlm:a.jpg",
        "save:a.jpg",
        "yolo:b.jpg",
        "vlm:b.jpg",
        "save:b.jpg",
    ]
    assert list((output_dir / "results").glob("*.json"))
    assert (output_dir / "batch_summary.json").exists()


def test_batch_summary_counts_retry_fallback_and_skipped() -> None:
    rows = [
        {
            "status": "success",
            "image_status": "completed",
            "image_name": "ok.jpg",
            "yolo_judgment": "OK",
            "yolo_detection_count": 0,
            "vlm_status": "not_run",
            "parse_status": "not_attempted",
            "fallback_used": "false",
            "retry_count": 0,
            "failure_reason": "",
            "total_processing_time_seconds": "1.000",
            "vlm_inference_time_seconds": "",
        },
        {
            "status": "success",
            "image_status": "completed",
            "image_name": "retry.jpg",
            "yolo_judgment": "NG",
            "yolo_detection_count": 1,
            "vlm_status": "retry_success",
            "parse_status": "success",
            "fallback_used": "false",
            "retry_count": 1,
            "failure_reason": "json_parse_failed",
            "total_processing_time_seconds": "2.000",
            "vlm_inference_time_seconds": "0.500",
        },
        {
            "status": "success",
            "image_status": "completed_with_fallback",
            "image_name": "fallback.jpg",
            "yolo_judgment": "NG",
            "yolo_detection_count": 1,
            "vlm_status": "done_false",
            "parse_status": "not_attempted",
            "fallback_used": "true",
            "retry_count": 2,
            "failure_reason": "done_false",
            "total_processing_time_seconds": "3.000",
            "vlm_inference_time_seconds": "1.000",
        },
    ]

    summary = batch._build_batch_summary(rows, [batch._summary_for_row(row) for row in rows], 6.0)

    assert summary["total_images"] == 3
    assert summary["ok_image_count"] == 1
    assert summary["ng_image_count"] == 2
    assert summary["vlm_executed_count"] == 2
    assert summary["vlm_skipped_count"] == 1
    assert summary["vlm_retry_success_count"] == 1
    assert summary["fallback_used_count"] == 1
    assert summary["done_false_count"] == 1
    assert summary["invalid_json_count"] == 1
    assert summary["average_image_processing_time_seconds"] == 2.0
    assert summary["average_vlm_processing_time_seconds"] == 0.75
