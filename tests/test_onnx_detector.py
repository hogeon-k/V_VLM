from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from model.defect_info import Detection
from service import onnx_detector
from service.onnx_detector import (
    LetterboxInfo,
    OnnxDetector,
    bbox_iou,
    class_aware_nms,
    detection_to_dict,
    letterbox,
    postprocess_output,
    register_windows_dll_directories,
    restore_boxes_to_original,
    validate_onnx_output,
    windows_dll_directory_candidates,
    xywh_to_xyxy,
)


def test_letterbox_result_shape() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    padded, info = letterbox(image, new_shape=960)

    assert padded.shape == (960, 960, 3)
    assert info.original_shape == (100, 200)
    assert info.new_unpad == (960, 480)
    assert info.pad == (0, 240)


def test_letterbox_coordinate_restore() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    _, info = letterbox(image, new_shape=960)
    original_box = np.array([[50, 20, 150, 80]], dtype=np.float32)
    letterboxed_box = original_box.copy()
    letterboxed_box[:, [0, 2]] = letterboxed_box[:, [0, 2]] * info.ratio[0] + info.pad[0]
    letterboxed_box[:, [1, 3]] = letterboxed_box[:, [1, 3]] * info.ratio[1] + info.pad[1]

    restored = restore_boxes_to_original(letterboxed_box, info)

    np.testing.assert_allclose(restored, original_box, atol=1e-4)


def test_xywh_to_xyxy() -> None:
    boxes = np.array([[100, 50, 20, 10]], dtype=np.float32)

    converted = xywh_to_xyxy(boxes)

    np.testing.assert_allclose(converted, np.array([[90, 45, 110, 55]], dtype=np.float32))


def test_bbox_iou() -> None:
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert bbox_iou([0, 0, 10, 10], [10, 10, 20, 20]) == pytest.approx(0.0)
    assert bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)


def test_class_aware_nms_keeps_different_classes() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, 1], dtype=np.int32)

    keep = class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)

    assert keep == [0, 1]


def test_class_aware_nms_suppresses_same_class_overlap() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, 0], dtype=np.int32)

    keep = class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)

    assert keep == [0]


def test_postprocess_empty_detection() -> None:
    output = np.zeros((1, 7, 2), dtype=np.float32)
    info = LetterboxInfo(
        original_shape=(100, 100),
        resized_shape=(960, 960),
        ratio=(9.6, 9.6),
        pad=(0, 0),
        new_unpad=(960, 960),
    )

    detections = postprocess_output(output, info, conf_threshold=0.15, iou_threshold=0.5)

    assert detections == []


def test_validate_onnx_output_shape() -> None:
    output = np.zeros((1, 7, 3), dtype=np.float32)

    validated = validate_onnx_output(output)

    assert validated.shape == (3, 7)


def test_validate_onnx_output_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        validate_onnx_output(np.zeros((7, 3), dtype=np.float32))


def test_detection_to_dict_structure() -> None:
    detection = Detection(1, "short", 0.8, 1, 2, 3, 4)

    data = detection_to_dict(detection)

    assert data == {
        "class_id": 1,
        "class_name": "short",
        "confidence": 0.8,
        "bbox": [1, 2, 3, 4],
    }


def test_windows_dll_directory_candidates_are_deduplicated(monkeypatch, tmp_path) -> None:
    torch_pkg = tmp_path / "torch"
    ort_pkg = tmp_path / "onnxruntime"
    monkeypatch.setattr(onnx_detector.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(onnx_detector, "package_subdir", lambda name, *parts: {"torch": torch_pkg / "lib", "onnxruntime": ort_pkg / "capi"}.get(name))

    candidates = windows_dll_directory_candidates()

    assert candidates == [
        tmp_path / "Lib" / "site-packages" / "torch" / "lib",
        tmp_path / "Lib" / "site-packages" / "onnxruntime" / "capi",
        torch_pkg / "lib",
        ort_pkg / "capi",
    ]


def test_register_windows_dll_directories_registers_existing_unique_paths(monkeypatch, tmp_path) -> None:
    first = tmp_path / "torch" / "lib"
    second = tmp_path / "onnxruntime" / "capi"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    handles: list[str] = []
    monkeypatch.setattr(onnx_detector.os, "name", "nt")
    monkeypatch.setattr(onnx_detector, "windows_dll_directory_candidates", lambda: [first, first, second, tmp_path / "missing"])
    monkeypatch.setattr(onnx_detector.os, "add_dll_directory", lambda path: handles.append(path) or object(), raising=False)
    onnx_detector._DLL_DIRECTORY_HANDLES.clear()

    registered = register_windows_dll_directories()
    registered_again = register_windows_dll_directories()

    assert registered == [first, second]
    assert registered_again == [first, second]
    assert handles == [str(first), str(second)]


def test_register_windows_dll_directories_noop_on_non_windows(monkeypatch) -> None:
    monkeypatch.setattr(onnx_detector.os, "name", "posix")

    assert register_windows_dll_directories() == []


class _FakeNode:
    def __init__(self, name: str, shape: list[int]) -> None:
        self.name = name
        self.shape = shape


class _FakeSession:
    def __init__(self, _model_path: str, providers: list[object]) -> None:
        self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if providers and providers[0] != "CPUExecutionProvider" else ["CPUExecutionProvider"]

    def get_providers(self) -> list[str]:
        return self._providers

    def get_inputs(self) -> list[_FakeNode]:
        return [_FakeNode("images", [1, 3, 960, 960])]

    def get_outputs(self) -> list[_FakeNode]:
        return [_FakeNode("output0", [1, 7, 18900])]


def _install_fake_onnxruntime(monkeypatch, session_cls: type[_FakeSession] = _FakeSession) -> None:
    fake_ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=session_cls,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)


def test_onnx_detector_cuda_session_success_sets_provider_state(monkeypatch, tmp_path) -> None:
    model = tmp_path / "best.onnx"
    model.write_bytes(b"fake")
    _install_fake_onnxruntime(monkeypatch)
    monkeypatch.setattr(onnx_detector, "register_windows_dll_directories", lambda: [tmp_path / "torch" / "lib"])
    monkeypatch.setattr(onnx_detector, "preload_torch_cuda_dlls", lambda: onnx_detector.TorchCudaPreloadResult(True, True))

    detector = OnnxDetector(model)
    detector._load_session()

    assert detector.using_cuda is True
    assert detector.actual_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert detector.cuda_preload_attempted is True
    assert detector.cuda_preload_success is True
    assert detector.registered_dll_directories == [str(tmp_path / "torch" / "lib")]


def test_onnx_detector_cpu_request_skips_torch_cuda_preload(monkeypatch, tmp_path) -> None:
    model = tmp_path / "best.onnx"
    model.write_bytes(b"fake")
    _install_fake_onnxruntime(monkeypatch)
    monkeypatch.setattr(onnx_detector, "preload_torch_cuda_dlls", lambda: pytest.fail("CPU mode must not preload torch CUDA"))

    detector = OnnxDetector(model, requested_provider="CPUExecutionProvider")
    detector._load_session()

    assert detector.using_cuda is False
    assert detector.cuda_preload_attempted is False


def test_onnx_detector_require_cuda_raises_on_fallback(monkeypatch, tmp_path) -> None:
    model = tmp_path / "best.onnx"
    model.write_bytes(b"fake")

    class CpuOnlySession(_FakeSession):
        def __init__(self, _model_path: str, providers: list[object]) -> None:
            self._providers = ["CPUExecutionProvider"]

    _install_fake_onnxruntime(monkeypatch, CpuOnlySession)
    monkeypatch.setattr(onnx_detector, "preload_torch_cuda_dlls", lambda: onnx_detector.TorchCudaPreloadResult(True, True))

    detector = OnnxDetector(model, require_cuda=True)
    with pytest.raises(RuntimeError, match="CPU-only session"):
        detector._load_session()
