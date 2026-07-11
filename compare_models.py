from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is normally installed with Ultralytics.
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "pcb" / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "compare"

# Optional edit-in-place defaults. CLI values always take precedence.
DEFAULT_MODEL_A: str | None = None
DEFAULT_MODEL_B: str | None = None

OVERALL_METRICS = ("Precision", "Recall", "mAP50", "mAP50-95")
SPEED_METRICS = ("preprocess speed", "inference speed", "postprocess speed")
CLASS_METRICS = ("Precision", "Recall", "AP50", "AP50-95", "GT objects")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate two YOLO models on the exact same dataset split and options, "
            "then compare overall, class-level, speed, and confusion-matrix metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A, help="First model weights path.")
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B, help="Second model weights path.")
    parser.add_argument("--name-a", default="default_model", help="Run folder name for model A.")
    parser.add_argument("--name-b", default="custom_model", help="Run folder name for model B.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=960, help="Validation image size.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold.")
    parser.add_argument("--device", default="0", help="Validation device, for example 0 or cpu.")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="Comparison output folder.")
    parser.add_argument("--split", default="val", help="Dataset split to validate.")
    return parser.parse_args()


def resolve_path(path_value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def list_best_pt_candidates(project_root: Path) -> list[Path]:
    detect_dir = project_root / "runs" / "detect"
    if not detect_dir.exists():
        return []
    return sorted(detect_dir.glob("*/weights/best.pt"))


def print_model_candidates(project_root: Path) -> None:
    candidates = list_best_pt_candidates(project_root)
    print("Available best.pt candidates under runs/detect:")
    if not candidates:
        print("  (none found)")
        return
    for candidate in candidates:
        print(f"  - {candidate.relative_to(project_root)}")


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read data.yaml. Install project requirements first.")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return data


def normalize_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, dict):
        return [str(raw_names[key]) for key in sorted(raw_names, key=lambda value: int(value))]
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    raise ValueError("data.yaml must contain names as a list or an index-to-name mapping.")


def dataset_base_dir(data_yaml: Path, data_config: dict[str, Any]) -> Path:
    raw_path = data_config.get("path")
    if raw_path is None:
        return data_yaml.parent
    base = Path(str(raw_path)).expanduser()
    if base.is_absolute():
        return base.resolve()

    project_relative = (PROJECT_ROOT / base).resolve()
    yaml_relative = (data_yaml.parent / base).resolve()
    if project_relative.exists() or not yaml_relative.exists():
        return project_relative
    return yaml_relative


def resolve_split_paths(data_yaml: Path, data_config: dict[str, Any], split: str) -> list[Path]:
    if split not in data_config:
        raise ValueError(f"data.yaml does not define split '{split}'.")
    split_value = data_config[split]
    values = split_value if isinstance(split_value, list) else [split_value]
    base_dir = dataset_base_dir(data_yaml, data_config)
    paths: list[Path] = []
    for value in values:
        split_path = Path(str(value)).expanduser()
        if not split_path.is_absolute():
            split_path = base_dir / split_path
        paths.append(split_path.resolve())
    return paths


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, dict[str, Any], list[str]]:
    if not args.model_a or not args.model_b:
        print_model_candidates(PROJECT_ROOT)
        raise ValueError("Both --model-a and --model-b are required. No model path is guessed automatically.")

    model_a = resolve_path(args.model_a)
    model_b = resolve_path(args.model_b)
    data_yaml = resolve_path(args.data)
    project_dir = resolve_path(args.project)

    if not model_a.is_file():
        raise FileNotFoundError(f"--model-a file does not exist: {model_a}")
    if not model_b.is_file():
        raise FileNotFoundError(f"--model-b file does not exist: {model_b}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"--data file does not exist: {data_yaml}")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0.")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1.")
    if not 0 <= args.iou <= 1:
        raise ValueError("--iou must be between 0 and 1.")

    data_config = load_yaml(data_yaml)
    class_names = normalize_names(data_config.get("names"))
    split_paths = resolve_split_paths(data_yaml, data_config, args.split)
    missing = [path for path in split_paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"data.yaml split '{args.split}' path does not exist: {missing_text}")

    return model_a, model_b, data_yaml, project_dir, data_config, class_names


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def first_existing_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def get_nested_metric(metrics: Any, attr_names: tuple[str, ...]) -> float | None:
    for container in (getattr(metrics, "box", None), metrics):
        if container is None:
            continue
        value = first_existing_attr(container, attr_names)
        if value is not None:
            return as_float(value() if callable(value) else value)
    return None


def per_class_array(metrics: Any, attr_names: tuple[str, ...], class_count: int) -> list[float | None]:
    for container in (getattr(metrics, "box", None), metrics):
        if container is None:
            continue
        value = first_existing_attr(container, attr_names)
        values = as_list(value() if callable(value) else value)
        if values:
            return pad_float_list(values, class_count)
    return [None] * class_count


def pad_float_list(values: list[Any], size: int) -> list[float | None]:
    result = [as_float(value) for value in values[:size]]
    result.extend([None] * (size - len(result)))
    return result


def class_index_map(metrics: Any, class_count: int) -> list[int]:
    box = getattr(metrics, "box", None)
    raw_indexes = first_existing_attr(box, ("ap_class_index", "class_result_index")) if box is not None else None
    indexes = as_list(raw_indexes)
    if indexes:
        parsed = []
        for index in indexes:
            try:
                parsed.append(int(index))
            except (TypeError, ValueError):
                pass
        if parsed:
            return parsed
    return list(range(class_count))


def align_class_values(values: list[float | None], metrics: Any, class_count: int) -> list[float | None]:
    indexes = class_index_map(metrics, class_count)
    if len(values) == class_count and indexes == list(range(class_count)):
        return values
    aligned: list[float | None] = [None] * class_count
    for position, class_index in enumerate(indexes):
        if 0 <= class_index < class_count and position < len(values):
            aligned[class_index] = values[position]
    return aligned


def extract_ap_values(metrics: Any, class_count: int) -> tuple[list[float | None], list[float | None]]:
    box = getattr(metrics, "box", None)
    all_ap = first_existing_attr(box, ("all_ap", "ap")) if box is not None else None
    rows = as_list(all_ap)
    if rows and isinstance(rows[0], list):
        ap50 = [as_float(row[0]) if row else None for row in rows]
        ap5095 = []
        for row in rows:
            row_values = [as_float(value) for value in row if as_float(value) is not None]
            ap5095.append(sum(row_values) / len(row_values) if row_values else None)
        return (
            align_class_values(pad_float_list(ap50, class_count), metrics, class_count),
            align_class_values(pad_float_list(ap5095, class_count), metrics, class_count),
        )

    ap50 = per_class_array(metrics, ("ap50", "ap_class_50"), class_count)
    ap5095 = per_class_array(metrics, ("ap", "ap_class", "maps"), class_count)
    return align_class_values(ap50, metrics, class_count), align_class_values(ap5095, metrics, class_count)


def extract_gt_counts(metrics: Any, class_count: int) -> list[int | None]:
    raw = first_existing_attr(metrics, ("nt_per_class", "nt", "targets_per_class", "seen_per_class"))
    values = as_list(raw)
    if values:
        result: list[int | None] = []
        for value in values[:class_count]:
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                result.append(None)
        result.extend([None] * (class_count - len(result)))
        return align_int_values(result, metrics, class_count)

    matrix_info = extract_confusion_matrix(metrics, class_count)
    if matrix_info.get("class_counts"):
        return matrix_info["class_counts"]
    return [None] * class_count


def align_int_values(values: list[int | None], metrics: Any, class_count: int) -> list[int | None]:
    indexes = class_index_map(metrics, class_count)
    if len(values) == class_count and indexes == list(range(class_count)):
        return values
    aligned: list[int | None] = [None] * class_count
    for position, class_index in enumerate(indexes):
        if 0 <= class_index < class_count and position < len(values):
            aligned[class_index] = values[position]
    return aligned


def extract_speed(metrics: Any) -> dict[str, float | None]:
    speed = getattr(metrics, "speed", {}) or {}
    if not isinstance(speed, dict):
        return {metric: None for metric in SPEED_METRICS}
    return {
        "preprocess speed": as_float(speed.get("preprocess")),
        "inference speed": as_float(speed.get("inference")),
        "postprocess speed": as_float(speed.get("postprocess")),
    }


def matrix_to_lists(matrix: Any) -> list[list[float]] | None:
    if matrix is None:
        return None
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    if not isinstance(matrix, list) or not matrix:
        return None
    rows: list[list[float]] = []
    for row in matrix:
        values = as_list(row)
        converted = [as_float(value) for value in values]
        if any(value is None for value in converted):
            return None
        rows.append([float(value) for value in converted if value is not None])
    return rows


def extract_confusion_matrix(metrics: Any, class_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "warning": None,
        "shape": None,
        "matrix": None,
        "tp_per_class": {},
        "fn_per_class": {},
        "fp_per_class": {},
        "total_fn": None,
        "total_fp": None,
        "class_confusions": {},
        "class_counts": None,
    }
    confusion = getattr(metrics, "confusion_matrix", None)
    matrix = matrix_to_lists(getattr(confusion, "matrix", None))
    if matrix is None:
        result["warning"] = "metrics.confusion_matrix.matrix was not available."
        return result

    row_count = len(matrix)
    col_count = len(matrix[0]) if matrix else 0
    result["shape"] = [row_count, col_count]
    result["matrix"] = matrix
    if any(len(row) != col_count for row in matrix):
        result["warning"] = "Confusion matrix rows have inconsistent lengths."
        return result
    if row_count < class_count or col_count < class_count:
        result["warning"] = f"Confusion matrix shape {row_count}x{col_count} is smaller than class count {class_count}."
        return result

    has_background_row = row_count == class_count + 1
    has_background_col = col_count == class_count + 1

    # Axis interpretation for this project: columns are True labels and rows are Predicted labels.
    # With an Ultralytics background row/column, matrix[-1][class] is FN and matrix[class][-1] is FP.
    tp = [matrix[index][index] for index in range(class_count)]
    fn = [matrix[-1][index] if has_background_row else 0.0 for index in range(class_count)]
    fp = [matrix[index][-1] if has_background_col else 0.0 for index in range(class_count)]
    class_counts = [int(round(sum(matrix[row][index] for row in range(row_count)))) for index in range(class_count)]

    confusions: dict[str, float] = {}
    total_confusion = 0.0
    for predicted in range(class_count):
        for actual in range(class_count):
            if predicted == actual:
                continue
            value = matrix[predicted][actual]
            if value:
                confusions[f"pred_{predicted}_true_{actual}"] = value
                total_confusion += value

    result.update(
        {
            "available": True,
            "tp_per_class": {str(index): tp[index] for index in range(class_count)},
            "fn_per_class": {str(index): fn[index] for index in range(class_count)},
            "fp_per_class": {str(index): fp[index] for index in range(class_count)},
            "total_fn": sum(fn),
            "total_fp": sum(fp),
            "class_confusions": confusions,
            "total_class_confusions": total_confusion,
            "class_counts": class_counts,
        }
    )
    return result


def extract_metrics(metrics: Any, class_names: list[str]) -> dict[str, Any]:
    class_count = len(class_names)
    precision = align_class_values(per_class_array(metrics, ("p", "precision"), class_count), metrics, class_count)
    recall = align_class_values(per_class_array(metrics, ("r", "recall"), class_count), metrics, class_count)
    ap50, ap5095 = extract_ap_values(metrics, class_count)
    gt_counts = extract_gt_counts(metrics, class_count)
    speed = extract_speed(metrics)
    confusion = extract_confusion_matrix(metrics, class_count)

    per_class = {}
    for index, class_name in enumerate(class_names):
        per_class[class_name] = {
            "Precision": precision[index],
            "Recall": recall[index],
            "AP50": ap50[index],
            "AP50-95": ap5095[index],
            "GT objects": gt_counts[index],
        }

    return {
        "overall": {
            "Precision": get_nested_metric(metrics, ("mp", "mean_precision", "precision")),
            "Recall": get_nested_metric(metrics, ("mr", "mean_recall", "recall")),
            "mAP50": get_nested_metric(metrics, ("map50", "map_50")),
            "mAP50-95": get_nested_metric(metrics, ("map", "map5095", "map50_95")),
            **speed,
        },
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def run_validation(
    model_path: Path,
    run_name: str,
    data_yaml: Path,
    project_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is not installed. Install project requirements before running validation.") from error

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        save_json=False,
        verbose=True,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )
    return {"raw": metrics, "save_dir": str(project_dir / run_name)}


def better_value(metric: str, value_a: float | int | None, value_b: float | int | None, name_a: str, name_b: str) -> str:
    a = as_float(value_a)
    b = as_float(value_b)
    if a is None or b is None:
        return "N/A"
    if a == b:
        return "Tie"
    lower_is_better = metric == "inference speed"
    if lower_is_better:
        return name_a if a < b else name_b
    return name_a if a > b else name_b


def format_value(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.6g}"


def print_overall_table(comparison: dict[str, dict[str, Any]], name_a: str, name_b: str) -> None:
    print("\nOverall comparison")
    print(f"{'Metric':<22}{name_a:<16}{name_b:<16}{'Better':<12}")
    print("-" * 66)
    for metric, row in comparison.items():
        print(
            f"{metric:<22}{format_value(row[name_a]):<16}"
            f"{format_value(row[name_b]):<16}{row['Better']:<12}"
        )


def print_class_table(rows: list[dict[str, Any]], name_a: str, name_b: str) -> None:
    print("\nClass comparison")
    print(f"{'Class':<18}{'Metric':<16}{name_a:<16}{name_b:<16}{'Better':<12}")
    print("-" * 78)
    for row in rows:
        print(
            f"{row['Class']:<18}{row['Metric']:<16}{format_value(row[name_a]):<16}"
            f"{format_value(row[name_b]):<16}{row['Better']:<12}"
        )


def build_overall_comparison(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    name_a: str,
    name_b: str,
) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}
    for metric in (*OVERALL_METRICS, *SPEED_METRICS):
        value_a = metrics_a["overall"].get(metric)
        value_b = metrics_b["overall"].get(metric)
        comparison[metric] = {
            name_a: value_a,
            name_b: value_b,
            "Better": better_value(metric, value_a, value_b, name_a, name_b),
        }
    return comparison


def build_class_comparison(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    class_names: list[str],
    name_a: str,
    name_b: str,
) -> list[dict[str, Any]]:
    rows = []
    for class_name in class_names:
        for metric in CLASS_METRICS:
            value_a = metrics_a["per_class"].get(class_name, {}).get(metric)
            value_b = metrics_b["per_class"].get(class_name, {}).get(metric)
            rows.append(
                {
                    "Class": class_name,
                    "Metric": metric,
                    name_a: value_a,
                    name_b: value_b,
                    "Better": better_value(metric, value_a, value_b, name_a, name_b),
                }
            )
    return rows


def missing_metrics(metrics_by_model: dict[str, dict[str, Any]], class_names: list[str]) -> list[str]:
    missing: list[str] = []
    for model_name, metrics in metrics_by_model.items():
        for metric, value in metrics["overall"].items():
            if value is None:
                missing.append(f"{model_name}.overall.{metric}")
        for class_name in class_names:
            class_metrics = metrics["per_class"].get(class_name, {})
            for metric, value in class_metrics.items():
                if value is None:
                    missing.append(f"{model_name}.class.{class_name}.{metric}")
        if not metrics["confusion_matrix"].get("available"):
            missing.append(f"{model_name}.confusion_matrix")
    return missing


def compare_gt_counts(metrics_a: dict[str, Any], metrics_b: dict[str, Any], class_names: list[str], name_a: str, name_b: str) -> bool:
    counts_a = [metrics_a["per_class"][class_name].get("GT objects") for class_name in class_names]
    counts_b = [metrics_b["per_class"][class_name].get("GT objects") for class_name in class_names]
    if any(count is None for count in counts_a + counts_b):
        print("[WARNING] Ground-truth object counts could not be fully extracted for both models.")
        return True

    total_a = sum(int(count) for count in counts_a if count is not None)
    total_b = sum(int(count) for count in counts_b if count is not None)
    valid = counts_a == counts_b and total_a == total_b
    if not valid:
        print("[ERROR] The two validation runs used different ground-truth object counts. The comparison is not valid.")
        print(f"Total GT objects: {name_a}={total_a}, {name_b}={total_b}")
        for class_name, count_a, count_b in zip(class_names, counts_a, counts_b):
            if count_a != count_b:
                print(f"  - {class_name}: {name_a}={count_a}, {name_b}={count_b}")
    return valid


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results(
    project_dir: Path,
    name_a: str,
    name_b: str,
    model_a: Path,
    model_b: Path,
    data_yaml: Path,
    args: argparse.Namespace,
    overall_comparison: dict[str, dict[str, Any]],
    class_rows: list[dict[str, Any]],
    metrics_by_model: dict[str, dict[str, Any]],
    gt_counts_match: bool,
    missing: list[str],
) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = [
        {"Metric": metric, name_a: row[name_a], name_b: row[name_b], "Better": row["Better"]}
        for metric, row in overall_comparison.items()
    ]
    write_csv(project_dir / "model_comparison.csv", overall_rows, ["Metric", name_a, name_b, "Better"])
    write_csv(project_dir / "class_comparison.csv", class_rows, ["Class", "Metric", name_a, name_b, "Better"])

    summary = {
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_a": str(model_a),
        "model_b": str(model_b),
        "data_yaml": str(data_yaml),
        "common_parameters": {
            "data": str(data_yaml),
            "split": args.split,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "plots": True,
            "save_json": False,
            "verbose": True,
        },
        "overall_metrics": metrics_by_model,
        "overall_comparison": overall_comparison,
        "class_comparison": class_rows,
        "better_by_metric": {metric: row["Better"] for metric, row in overall_comparison.items()},
        "ground_truth_counts_match": gt_counts_match,
        "missing_metrics": missing,
    }
    with (project_dir / "comparison_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def print_command_example() -> None:
    print("\nExample command")
    print("python compare_models.py `")
    print("  --model-a runs\\detect\\pcb_default\\weights\\best.pt `")
    print("  --model-b runs\\detect\\pcb_custom\\weights\\best.pt `")
    print("  --name-a default `")
    print("  --name-b custom `")
    print("  --data datasets\\pcb\\data.yaml `")
    print("  --imgsz 960 `")
    print("  --conf 0.001 `")
    print("  --iou 0.7 `")
    print("  --device 0 `")
    print("  --split val")


def main() -> int:
    args = parse_args()
    try:
        model_a, model_b, data_yaml, project_dir, _data_config, class_names = validate_inputs(args)
        print_model_candidates(PROJECT_ROOT)
        print(f"\nValidating {args.name_a}: {model_a}")
        run_a = run_validation(model_a, args.name_a, data_yaml, project_dir, args)
        print(f"\nValidating {args.name_b}: {model_b}")
        run_b = run_validation(model_b, args.name_b, data_yaml, project_dir, args)

        metrics_a = extract_metrics(run_a["raw"], class_names)
        metrics_b = extract_metrics(run_b["raw"], class_names)
        metrics_a["save_dir"] = run_a["save_dir"]
        metrics_b["save_dir"] = run_b["save_dir"]

        overall_comparison = build_overall_comparison(metrics_a, metrics_b, args.name_a, args.name_b)
        class_rows = build_class_comparison(metrics_a, metrics_b, class_names, args.name_a, args.name_b)
        metrics_by_model = {args.name_a: metrics_a, args.name_b: metrics_b}
        missing = missing_metrics(metrics_by_model, class_names)
        gt_counts_match = compare_gt_counts(metrics_a, metrics_b, class_names, args.name_a, args.name_b)

        for model_name, metrics in metrics_by_model.items():
            warning = metrics["confusion_matrix"].get("warning")
            if warning:
                print(f"[WARNING] {model_name}: {warning}")

        print_overall_table(overall_comparison, args.name_a, args.name_b)
        print_class_table(class_rows, args.name_a, args.name_b)
        save_results(
            project_dir,
            args.name_a,
            args.name_b,
            model_a,
            model_b,
            data_yaml,
            args,
            overall_comparison,
            class_rows,
            metrics_by_model,
            gt_counts_match,
            missing,
        )
        print(f"\nSaved results to: {project_dir}")
        print_command_example()
        return 2 if not gt_counts_match else 0
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        if isinstance(error, (ValueError, FileNotFoundError)):
            print_model_candidates(PROJECT_ROOT)
            print_command_example()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
