from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.evaluate_onnx import (
    ap_for_threshold,
    best_candidate_for_gt,
    best_debug_candidate,
    classify_debug_stage,
    compare_backends,
    compute_map,
    evaluate_predictions,
    g1_judgement,
    match_float_predictions,
    prediction_record,
    resolve_dataset_paths,
    speed_comparison_validity,
)
from compare_predictions import Detection


def test_ap_for_threshold_handles_empty_predictions() -> None:
    gt = {"image.jpg": [Detection(0, (0, 0, 10, 10))]}

    assert ap_for_threshold([], gt, gt_count=1, threshold=0.5) == 0.0


def test_compute_map_uses_same_class_one_to_one_matching() -> None:
    gt = {"image.jpg": [Detection(0, (0, 0, 10, 10)), Detection(0, (20, 20, 30, 30))]}
    predictions = {
        "image.jpg": [
            Detection(0, (0, 0, 10, 10), 0.9),
            Detection(1, (20, 20, 30, 30), 0.8),
            Detection(0, (20, 20, 30, 30), 0.7),
        ]
    }

    ap = compute_map(predictions, gt, ["open_circuit", "short"], (0.5,))

    assert ap["open_circuit"][0.5] > 0.95
    assert ap["short"][0.5] == 0.0


def test_evaluate_predictions_counts_tp_fp_fn() -> None:
    gt = {"image.jpg": [Detection(0, (0, 0, 10, 10)), Detection(1, (20, 20, 30, 30))]}
    predictions = {"image.jpg": [Detection(0, (0, 0, 10, 10), 0.9), Detection(1, (40, 40, 50, 50), 0.8)]}

    metrics, records = evaluate_predictions("onnx", predictions, gt, ["open_circuit", "short"], 0.5, (0.5,), {})

    assert metrics["overall"]["tp"] == 1
    assert metrics["overall"]["fp"] == 1
    assert metrics["overall"]["fn"] == 1
    assert {record["error_type"] for record in records} == {"TP", "FP", "FN"}


def test_compare_backends_warning_thresholds() -> None:
    args = argparse.Namespace(
        match_iou=0.5,
        fail_map50_diff=0.01,
        fail_map5095_diff=0.01,
        fail_precision_diff=0.02,
        fail_recall_diff=0.02,
        min_avg_matched_box_iou=0.99,
        max_new_fp=0,
        max_new_fn=0,
    )
    pt_metrics = {
        "overall": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "mAP50": 1.0, "mAP50-95": 1.0, "tp": 1, "fp": 0, "fn": 0},
        "per_class": {"short": {"AP50": 1.0, "AP50-95": 1.0}},
    }
    onnx_metrics = {
        "overall": {"precision": 0.5, "recall": 1.0, "f1": 0.67, "mAP50": 0.5, "mAP50-95": 0.5, "tp": 1, "fp": 1, "fn": 0},
        "per_class": {"short": {"AP50": 0.5, "AP50-95": 0.5}},
    }
    pt_predictions = {"image.jpg": [Detection(1, (0, 0, 10, 10), 0.9)]}
    onnx_predictions = {
        "image.jpg": [
            Detection(1, (0, 0, 10, 10), 0.9),
            Detection(1, (40, 40, 50, 50), 0.8),
        ]
    }

    comparison, rows = compare_backends(pt_metrics, onnx_metrics, pt_predictions, onnx_predictions, args)

    assert comparison["overall_result"] == "WARNING"
    assert comparison["new_fp_count"] == 1
    assert rows[0]["onnx_only"] == 1


def test_float_bbox_prevents_rounding_boundary_fn() -> None:
    pt = [Detection(0, (2088.6489, 999.6395, 2160.3096, 1062.0406), 0.078)]
    onnx = [Detection(0, (2088.6487, 999.6379, 2160.3174, 1062.0441), 0.079)]
    rounded_onnx = [Detection(0, tuple(float(v) for v in (2089, 1000, 2160, 1062)), 0.079)]

    assert match_float_predictions(pt, onnx, 0.999)[0]["status"] == "MATCHED"
    assert match_float_predictions(pt, rounded_onnx, 0.999)[0]["status"] == "PT_ONLY"


def test_prediction_record_separates_evaluation_and_display_bbox() -> None:
    record = prediction_record(Detection(0, (1.2, 2.6, 3.4, 4.8), 0.5), ["open_circuit"])

    assert record["evaluation_bbox"] == [1.2, 2.6, 3.4, 4.8]
    assert record["display_bbox"] == [1, 3, 3, 5]


def test_speed_comparison_invalid_for_gpu_vs_cpu() -> None:
    valid, reason = speed_comparison_validity("cuda", "CPUExecutionProvider")

    assert valid is False
    assert "PyTorch used cuda" in reason


def test_best_candidate_for_gt_uses_class_when_requested() -> None:
    gt = Detection(0, (0, 0, 10, 10))
    candidates = [Detection(1, (0, 0, 10, 10), 0.9), Detection(0, (20, 20, 30, 30), 0.8)]

    assert best_candidate_for_gt(gt, candidates, same_class=True).class_id == 0
    assert best_candidate_for_gt(gt, candidates, same_class=False).class_id == 1


def test_debug_stage_classifies_confidence_filter_removal() -> None:
    case = {"gt": {"bbox": [0, 0, 10, 10], "class_id": 0}}

    result = classify_debug_stage([case], [], [], [], ["short"], 0.5)

    assert result["classification"] == "removed_by_confidence_filter_or_absent_in_raw_output"


def test_debug_stage_classifies_nms_removal() -> None:
    case = {"gt": {"bbox": [0, 0, 10, 10], "class_id": 0}}
    after_conf = [{"class_id": 0, "bbox_xyxy_original": [0, 0, 10, 10], "confidence": 0.9}]

    result = classify_debug_stage([case], after_conf, [], [], ["short"], 0.5)

    assert result["classification"] == "removed_by_nms"


def test_debug_stage_classifies_match_iou_shortfall() -> None:
    case = {"gt": {"bbox": [0, 0, 10, 10], "class_id": 0}}
    kept = [{"class_id": 0, "bbox_xyxy_original": [20, 20, 30, 30], "confidence": 0.9}]

    result = classify_debug_stage([case], kept, kept, [], ["short"], 0.5)

    assert result["classification"] == "kept_after_nms_but_match_iou_below_threshold"


def test_g1_judgement_conditional_pass_when_metrics_within_limits() -> None:
    final_payload = {
        "overall_result": "WARNING",
        "differences": {"mAP50": 0.001, "mAP50-95": 0.001, "precision": 0.001, "recall": 0.001},
        "thresholds": {"mAP50_difference": 0.01, "mAP50_95_difference": 0.01, "precision_difference": 0.02, "recall_difference": 0.02},
    }

    assert g1_judgement(final_payload, {"new_fn_count": 1}) == "CONDITIONAL PASS"


def test_resolve_dataset_paths_project_relative(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "datasets" / "pcb"
    (dataset / "images" / "test").mkdir(parents=True)
    (dataset / "labels" / "test").mkdir(parents=True)
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text("path: datasets/pcb\ntest: images/test\nnames:\n  0: short\n", encoding="utf-8")
    monkeypatch.setattr("scripts.evaluate_onnx.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("compare_predictions.PROJECT_ROOT", tmp_path)

    images, labels, names = resolve_dataset_paths(data_yaml, "test")

    assert images == dataset / "images" / "test"
    assert labels == dataset / "labels" / "test"
    assert names == ["short"]
