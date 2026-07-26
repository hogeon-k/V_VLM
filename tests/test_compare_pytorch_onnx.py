from __future__ import annotations

import argparse
from pathlib import Path

from model.defect_info import Detection
from scripts.compare_pytorch_onnx import (
    build_report,
    match_detections,
    summarize_accuracy,
)


class DummyOnnxTimed:
    detections = [Detection(1, "short", 0.89, 0, 0, 10, 10)]
    providers = ["CPUExecutionProvider"]
    input_name = "images"
    output_name = "output0"
    input_shape = [1, 3, 960, 960]
    output_shape = [1, 7, 1]


def test_match_detections_by_same_class_and_best_iou() -> None:
    pt = [
        Detection(1, "short", 0.9, 0, 0, 10, 10),
        Detection(0, "open_circuit", 0.7, 50, 50, 60, 60),
    ]
    onnx = [
        Detection(1, "short", 0.89, 1, 1, 11, 11),
        Detection(0, "open_circuit", 0.7, 80, 80, 90, 90),
    ]

    matches = match_detections(pt, onnx, match_iou=0.5)

    assert [match.status for match in matches] == ["MATCHED", "PT_ONLY", "ONNX_ONLY"]


def test_match_detections_does_not_match_different_classes() -> None:
    pt = [Detection(1, "short", 0.9, 0, 0, 10, 10)]
    onnx = [Detection(2, "missing_hole", 0.9, 0, 0, 10, 10)]

    matches = match_detections(pt, onnx, match_iou=0.5)

    assert [match.status for match in matches] == ["PT_ONLY", "ONNX_ONLY"]


def test_empty_detection_summary() -> None:
    summary, warnings = summarize_accuracy([])

    assert summary["matched_count"] == 0
    assert summary["pt_only_count"] == 0
    assert summary["onnx_only_count"] == 0
    assert warnings == []


def test_json_report_structure(monkeypatch) -> None:
    args = argparse.Namespace(
        pt_model=Path("models/best.pt"),
        onnx_model=Path("models/best.onnx"),
        imgsz=960,
        conf=0.15,
        iou=0.5,
        device="0",
        warmup=5,
        runs=20,
    )
    image_path = Path("data/images/01_short_01.jpg")

    class FakeImage:
        shape = (100, 200, 3)

    monkeypatch.setattr("scripts.compare_pytorch_onnx.cv2.imread", lambda _: FakeImage())
    pt = [Detection(1, "short", 0.9, 0, 0, 10, 10)]
    matches = match_detections(pt, DummyOnnxTimed.detections)
    accuracy, warnings = summarize_accuracy(matches)
    timing = {
        "preprocess_ms": {"avg": 1.0, "min": 1.0, "max": 1.0, "median": 1.0, "stdev": 0.0},
        "inference_ms": {"avg": 2.0, "min": 2.0, "max": 2.0, "median": 2.0, "stdev": 0.0},
        "postprocess_ms": {"avg": 1.0, "min": 1.0, "max": 1.0, "median": 1.0, "stdev": 0.0},
        "total_ms": {"avg": 4.0, "min": 4.0, "max": 4.0, "median": 4.0, "stdev": 0.0},
    }

    report = build_report(args, image_path, {1: "short"}, pt, DummyOnnxTimed(), timing, timing, matches, accuracy, warnings)

    assert report["models"]["pytorch"] == "models\\best.pt" or report["models"]["pytorch"] == "models/best.pt"
    assert report["onnx_runtime"]["input_name"] == "images"
    assert report["detections"]["pytorch"][0]["bbox"] == [0, 0, 10, 10]
    assert report["judgement"]["status"] in {"PASS", "WARNING"}
