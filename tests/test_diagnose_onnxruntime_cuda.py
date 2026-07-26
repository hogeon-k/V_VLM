from __future__ import annotations

import json

from scripts.diagnose_onnxruntime_cuda import cuda_session_is_valid, diagnose_summary, write_json


def test_diagnose_summary_detects_duplicate_runtime_packages(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.diagnose_onnxruntime_cuda.package_version",
        lambda name: "1.0" if name in {"onnxruntime", "onnxruntime-gpu"} else None,
    )

    assert "Both onnxruntime and onnxruntime-gpu" in diagnose_summary()


def test_diagnose_summary_detects_missing_gpu_package(monkeypatch) -> None:
    monkeypatch.setattr("scripts.diagnose_onnxruntime_cuda.package_version", lambda _name: None)

    assert "onnxruntime-gpu is not installed" in diagnose_summary()


def test_cuda_session_is_valid_requires_cuda_provider_and_clean_inference() -> None:
    diagnostics = {
        "onnxruntime_prepared_session": {
            "session_creation": {"actual_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]},
            "test_inference": {"status": "PASS", "valid": True},
        }
    }

    assert cuda_session_is_valid(diagnostics) is True

    diagnostics["onnxruntime_prepared_session"]["session_creation"]["actual_providers"] = ["CPUExecutionProvider"]
    assert cuda_session_is_valid(diagnostics) is False


def test_write_json_uses_utf8_and_pretty_format(tmp_path) -> None:
    output = tmp_path / "diagnostics.json"

    write_json(output, {"diagnosis": "CUDA 진단", "ok": True})

    saved = output.read_text(encoding="utf-8")
    assert "\n  " in saved
    assert json.loads(saved)["diagnosis"] == "CUDA 진단"
