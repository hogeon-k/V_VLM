from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.onnx_detector import DEFAULT_CLASS_NAMES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ONNX model and optionally write model metadata.")
    parser.add_argument("--model", type=Path, default=Path("models/best.onnx"))
    parser.add_argument("--source-model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("datasets/pcb/data.yaml"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/onnx/onnx_validation.json"))
    parser.add_argument("--metadata-output", type=Path, default=Path("models/model_metadata.json"))
    parser.add_argument("--write-metadata", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def load_class_names(data_yaml: Path) -> dict[int, str]:
    if not data_yaml.exists():
        return DEFAULT_CLASS_NAMES
    try:
        import yaml

        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        return DEFAULT_CLASS_NAMES
    names = data.get("names", {}) if isinstance(data, dict) else {}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return DEFAULT_CLASS_NAMES


def tensor_shape(value_info: Any) -> list[Any]:
    dims = []
    shape = value_info.type.tensor_type.shape
    for dim in shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return dims


def has_dynamic_shape(shapes: list[list[Any]]) -> bool:
    return any(not isinstance(dim, int) for shape in shapes for dim in shape)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model(model_path: Path) -> tuple[dict[str, Any], int]:
    model_path = resolve_project_path(model_path)
    report: dict[str, Any] = {
        "model": relative_or_absolute(model_path),
        "exists": model_path.is_file(),
        "loaded": False,
        "checker_passed": False,
        "error": None,
    }
    if not model_path.is_file():
        report["error"] = f"ONNX model file does not exist: {model_path}"
        return report, 1

    try:
        import onnx
    except ImportError:
        report["error"] = "The 'onnx' package is not installed. Install requirements.txt and try again."
        return report, 1

    try:
        model = onnx.load(str(model_path))
        report["loaded"] = True
        onnx.checker.check_model(model)
        report["checker_passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report, 1

    inputs = [{"name": item.name, "shape": tensor_shape(item)} for item in model.graph.input]
    outputs = [{"name": item.name, "shape": tensor_shape(item)} for item in model.graph.output]
    opsets = [{"domain": item.domain or "ai.onnx", "version": int(item.version)} for item in model.opset_import]
    shapes = [item["shape"] for item in [*inputs, *outputs]]
    report.update(
        {
            "inputs": inputs,
            "outputs": outputs,
            "opsets": opsets,
            "opset": opsets[0]["version"] if opsets else None,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "dynamic_shape": has_dynamic_shape(shapes),
            "file_size_bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        }
    )
    return report, 0


def build_metadata(validation: dict[str, Any], source_model: Path, data_yaml: Path) -> dict[str, Any]:
    class_names = load_class_names(resolve_project_path(data_yaml))
    input_shape = validation.get("inputs", [{}])[0].get("shape", [])
    output_names = [item["name"] for item in validation.get("outputs", [])]
    input_size = input_shape[2:4] if len(input_shape) >= 4 and all(isinstance(value, int) for value in input_shape[2:4]) else []
    return {
        "model_name": Path(validation.get("model", "best.onnx")).name,
        "source_model": relative_or_absolute(resolve_project_path(source_model)),
        "task": "detect",
        "input_name": validation.get("inputs", [{}])[0].get("name"),
        "input_shape": input_shape,
        "input_size": input_size,
        "batch_size": input_shape[0] if input_shape and isinstance(input_shape[0], int) else None,
        "dynamic": bool(validation.get("dynamic_shape")),
        "opset": validation.get("opset"),
        "class_count": len(class_names),
        "class_names": [class_names[index] for index in sorted(class_names)],
        "output_names": output_names,
        "onnx_file_size_bytes": validation.get("file_size_bytes"),
        "onnx_sha256": validation.get("sha256"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_json(path: Path, data: Any) -> None:
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    print("=== ONNX Model Validation ===")
    print(f"Model: {report['model']}")
    print(f"Checker result: {'PASS' if report.get('checker_passed') else 'FAIL'}")
    for item in report.get("inputs", []):
        print(f"Input: {item['name']} {item['shape']}")
    for item in report.get("outputs", []):
        print(f"Output: {item['name']} {item['shape']}")
    print(f"Opset: {report.get('opset')}")
    print(f"Dynamic shape: {report.get('dynamic_shape')}")
    print(f"File size: {report.get('file_size_bytes')}")
    if report.get("error"):
        print(f"ERROR: {report['error']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report, exit_code = validate_model(args.model)
    write_json(args.output, report)
    if exit_code == 0 and args.write_metadata:
        write_json(args.metadata_output, build_metadata(report, args.source_model, args.data))
    print_summary(report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
