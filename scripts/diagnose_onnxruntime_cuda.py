from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.onnx_detector import preload_torch_cuda_dlls, register_windows_dll_directories


CUDA_DLL_NAMES = (
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "onnxruntime_providers_cuda.dll",
    "zlibwapi.dll",
    "msvcp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose ONNX Runtime CUDAExecutionProvider availability.")
    parser.add_argument("--model", type=Path, default=Path("models/best.onnx"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/onnx/cuda_diagnostics.json"))
    parser.add_argument("--require-cuda", action="store_true", help="Return a non-zero exit code if the session falls back to CPU.")
    return parser.parse_args(argv)


def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_command(args: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
        return {
            "command": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"command": args, "error": f"{type(exc).__name__}: {exc}"}


def path_entries() -> list[str]:
    return [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]


def find_related_path_entries() -> list[str]:
    keywords = ("cuda", "cudnn", "nvidia", "onnxruntime")
    return [entry for entry in path_entries() if any(keyword in entry.lower() for keyword in keywords)]


def dll_presence() -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {name: [] for name in CUDA_DLL_NAMES}
    search_dirs = [Path(entry) for entry in path_entries()]
    try:
        import onnxruntime as ort

        search_dirs.append(Path(ort.__file__).resolve().parent / "capi")
    except Exception:
        pass
    try:
        import torch

        search_dirs.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        pass
    system32 = Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32"
    search_dirs.append(system32)
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for name in CUDA_DLL_NAMES:
            candidate = directory / name
            if candidate.exists():
                candidates[name].append(str(candidate))
    return candidates


def cuda_environment_variables() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.upper().startswith("CUDA_PATH") or "CUDNN" in key.upper()
    }


def torch_lib_path() -> str | None:
    try:
        import torch

        return str(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        return None


def torch_info() -> dict[str, Any]:
    data: dict[str, Any] = {}
    try:
        import torch

        data["version"] = torch.__version__
        data["cuda_available"] = bool(torch.cuda.is_available())
        data["torch_cuda_version"] = torch.version.cuda
        data["cudnn_version"] = torch.backends.cudnn.version()
        data["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        data["error"] = f"{type(exc).__name__}: {exc}"
    return data


def onnxruntime_info(model_path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "onnxruntime_version": package_version("onnxruntime"),
        "onnxruntime_gpu_version": package_version("onnxruntime-gpu"),
        "requested_providers": [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        "session_creation": {"requested_provider": "CUDAExecutionProvider"},
    }
    try:
        import onnxruntime as ort

        data["imported_from"] = str(Path(ort.__file__).resolve())
        data["runtime_version"] = ort.__version__
        data["available_providers"] = ort.get_available_providers()
        data["all_providers"] = ort.get_all_providers()
        try:
            session = ort.InferenceSession(str(model_path), providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
            actual_providers = session.get_providers()
            status = "PASS" if "CUDAExecutionProvider" in actual_providers else "FALLBACK"
            data["session_creation"].update({"status": status, "actual_providers": actual_providers})
            data["test_inference"] = run_dummy_inference(session)
        except Exception as exc:
            data["session_creation"].update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
            try:
                fallback = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                data["cpu_fallback"] = {"status": "PASS", "actual_providers": fallback.get_providers()}
            except Exception as fallback_exc:
                data["cpu_fallback"] = {"status": "FAIL", "error": f"{type(fallback_exc).__name__}: {fallback_exc}"}
    except Exception as exc:
        data["import_error"] = f"{type(exc).__name__}: {exc}"
    return data


def prepared_onnxruntime_info(model_path: Path, *, preload_torch_cuda: bool = True, register_windows_dlls: bool = True) -> dict[str, Any]:
    registered_dirs = register_windows_dll_directories() if register_windows_dlls else []
    preload = preload_torch_cuda_dlls() if preload_torch_cuda else None
    data = onnxruntime_info(model_path)
    data["registered_dll_directories"] = [str(path) for path in registered_dirs]
    data["torch_cuda_preload"] = {
        "attempted": bool(preload.attempted) if preload else False,
        "success": bool(preload.success) if preload else False,
        "error": preload.error if preload else None,
        "cuda_available": preload.cuda_available if preload else None,
        "torch_version": preload.torch_version if preload else None,
        "torch_cuda_version": preload.torch_cuda_version if preload else None,
        "cudnn_version": preload.cudnn_version if preload else None,
    }
    return data


def run_dummy_inference(session: Any) -> dict[str, Any]:
    try:
        import numpy as np

        model_input = session.get_inputs()[0]
        shape = [int(dim) if isinstance(dim, int) else 1 for dim in model_input.shape]
        tensor = np.zeros(shape, dtype=np.float32)
        outputs = session.run(None, {model_input.name: tensor})
        has_nan = any(bool(np.isnan(output).any()) for output in outputs)
        has_inf = any(bool(np.isinf(output).any()) for output in outputs)
        return {
            "status": "PASS",
            "input_name": model_input.name,
            "input_shape": shape,
            "output_shapes": [list(output.shape) for output in outputs],
            "has_nan": has_nan,
            "has_inf": has_inf,
            "valid": not has_nan and not has_inf,
        }
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def build_diagnostics(model_path: Path) -> dict[str, Any]:
    model_path = resolve_project_path(model_path)
    prepared_ort = prepared_onnxruntime_info(model_path)
    torch = torch_info()
    dlls = dll_presence()
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(model_path),
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            "onnx": package_version("onnx"),
            "onnxruntime": package_version("onnxruntime"),
            "onnxruntime-gpu": package_version("onnxruntime-gpu"),
            "torch": package_version("torch"),
        },
        "onnxruntime_prepared_session": prepared_ort,
        "torch": torch,
        "nvidia_smi": run_command(["nvidia-smi"]),
        "nvcc": run_command(["nvcc", "--version"]),
        "path_cuda_related_entries": find_related_path_entries(),
        "cuda_environment_variables": cuda_environment_variables(),
        "torch_lib_path": torch_lib_path(),
        "dll_presence": dlls,
        "recommended_powershell": recommended_powershell(dlls),
        "duplicate_runtime_packages": {
            "onnxruntime_and_gpu_both_installed": package_version("onnxruntime") is not None and package_version("onnxruntime-gpu") is not None
        },
        "diagnosis": diagnose_summary(),
    }


def recommended_powershell(dlls: dict[str, list[str]]) -> dict[str, Any]:
    required = ("cudnn64_9.dll", "cublas64_12.dll", "cudart64_12.dll", "onnxruntime_providers_cuda.dll")
    dirs: list[str] = []
    for name in required:
        paths = dlls.get(name, [])
        if not paths:
            continue
        parent = str(Path(paths[0]).parent)
        if parent not in dirs:
            dirs.append(parent)
    if not dirs:
        return {"available": False, "reason": "No complete candidate DLL directories were found."}
    path_prefix = ";".join(dirs)
    return {
        "available": True,
        "session_only_path_command": f'$env:PATH = "{path_prefix};$env:PATH"',
        "verify_command": r".\.venv\Scripts\python.exe scripts\diagnose_onnxruntime_cuda.py --model models\best.onnx --output benchmarks\onnx\cuda_diagnostics_after_path.json",
        "note": "Apply only in the current PowerShell session first. Do not make permanent PATH changes until diagnostics pass.",
    }


def diagnose_summary() -> str:
    ort = package_version("onnxruntime")
    ort_gpu = package_version("onnxruntime-gpu")
    if ort and ort_gpu:
        return "Both onnxruntime and onnxruntime-gpu are installed; prefer a single ONNX Runtime package in the environment."
    if not ort_gpu:
        return "onnxruntime-gpu is not installed, so CUDAExecutionProvider cannot be used."
    return "Review CUDA/cuDNN/MSVC DLL availability and ONNX Runtime GPU version requirements."


def cuda_session_is_valid(data: dict[str, Any]) -> bool:
    ort_info = data.get("onnxruntime_prepared_session", {})
    session = ort_info.get("session_creation", {})
    inference = ort_info.get("test_inference", {})
    return (
        "CUDAExecutionProvider" in (session.get("actual_providers") or [])
        and inference.get("status") == "PASS"
        and inference.get("valid") is True
    )


def print_summary(data: dict[str, Any]) -> None:
    ort_info = data.get("onnxruntime_prepared_session", {})
    session = ort_info.get("session_creation", {})
    inference = ort_info.get("test_inference", {})
    preload = ort_info.get("torch_cuda_preload", {})
    print("=== ONNX Runtime CUDA Diagnostics ===")
    print(f"Python: {data['python'].split()[0]}")
    print(f"Python executable: {data.get('python_executable')}")
    print(f"sys.prefix: {data.get('sys_prefix')}")
    print(f"onnxruntime-gpu: {data['packages'].get('onnxruntime-gpu')}")
    print(f"Available providers: {ort_info.get('available_providers')}")
    print(f"Registered DLL directories: {ort_info.get('registered_dll_directories')}")
    print(f"Torch CUDA preload: {'PASS' if preload.get('success') else 'FAIL'}")
    print(f"Requested providers: {ort_info.get('requested_providers')}")
    print(f"Actual providers: {session.get('actual_providers') or ort_info.get('cpu_fallback', {}).get('actual_providers')}")
    print(f"CUDA session creation: {session.get('status')}")
    if inference:
        print(f"CUDA test inference: {inference.get('status')}")
        print(f"Output shapes: {inference.get('output_shapes')}")
        print(f"NaN/Inf: {inference.get('has_nan')} / {inference.get('has_inf')}")
    if session.get("error"):
        print(f"Reason: {session['error']}")
    elif inference.get("error"):
        print(f"Reason: {inference['error']}")


def write_json(path: Path, data: Any) -> None:
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    diagnostics = build_diagnostics(args.model)
    write_json(args.output, diagnostics)
    print_summary(diagnostics)
    if args.require_cuda and not cuda_session_is_valid(diagnostics):
        print(
            "CUDAExecutionProvider was required, but ONNX Runtime did not create a valid CUDA session. "
            "See the JSON diagnostics for details."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
