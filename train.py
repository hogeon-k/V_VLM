from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"
DEFAULT_DATA_PATH = PROJECT_ROOT / "datasets" / "pcb" / "data.yaml"
DEFAULT_PROJECT_DIR = PROJECT_ROOT / "runs" / "detect"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO11 detection model for the PCB defect dataset.",
    )
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL_PATH), help="Model weights path."
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA_PATH), help="YOLO dataset YAML path."
    )
    parser.add_argument(
        "--project-name", default="pcb_yolo11n", help="Run name under runs/detect."
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs."
    )
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Training device. Use 0, 1, cpu, etc. If omitted, CUDA is used when available.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Data loader worker count."
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="Early stopping patience."
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Checkpoint save interval in epochs. Use -1 to disable periodic checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted Ultralytics run from a last.pt checkpoint.",
    )
    parser.add_argument(
        "--cache", action="store_true", help="Cache dataset images during training."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the resolved training configuration without training.",
    )
    return parser.parse_args()


def resolve_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def choose_device(requested_device: str | None) -> int | str:
    if requested_device:
        device = requested_device.strip()
        if device.isdigit():
            return int(device)
        return device
    return 0 if torch.cuda.is_available() else "cpu"


def read_dataset_yaml(data_path: Path) -> dict[str, Any]:
    try:
        with data_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Dataset YAML is not valid YAML: {data_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_path}")
    return data


def resolve_dataset_split_path(
    data_path: Path, dataset: dict[str, Any], split: str
) -> Path:
    split_value = dataset.get(split)
    if not split_value:
        raise ValueError(
            f"Dataset YAML is missing required '{split}' entry: {data_path}"
        )

    split_path = Path(str(split_value))
    if split_path.is_absolute():
        return split_path.resolve()

    dataset_root_value = dataset.get("path")
    if dataset_root_value:
        dataset_root = Path(str(dataset_root_value))
        if not dataset_root.is_absolute():
            dataset_root = PROJECT_ROOT / dataset_root
    else:
        dataset_root = data_path.parent

    return (dataset_root / split_path).resolve()


def validate_training_inputs(
    args: argparse.Namespace, model_path: Path, data_path: Path
) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"[ERROR] Model file not found:\n{model_path}")

    if not data_path.is_file():
        raise FileNotFoundError(f"[ERROR] Dataset YAML not found:\n{data_path}")

    dataset = read_dataset_yaml(data_path)
    for split in ("train", "val"):
        split_path = resolve_dataset_split_path(data_path, dataset, split)
        if not split_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Dataset '{split}' path from data.yaml was not found:\n{split_path}"
            )

    positive_values = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(
                f"[ERROR] --{name} must be a positive integer, got {value}."
            )

    if args.patience < 0:
        raise ValueError(
            f"[ERROR] --patience must be 0 or greater, got {args.patience}."
        )

    if args.save_period == 0 or args.save_period < -1:
        raise ValueError("[ERROR] --save-period must be -1 or a positive integer.")

    if args.resume and model_path.name != "last.pt":
        raise ValueError(
            "[ERROR] --resume should point --model to an Ultralytics last.pt checkpoint.\n"
            "Example: python train.py --resume --model runs/detect/pcb_yolo11n/weights/last.pt"
        )


def print_training_summary(
    args: argparse.Namespace,
    model_path: Path,
    data_path: Path,
    output_dir: Path,
    device: int | str,
) -> None:
    print("Training configuration")
    print(f"  Project root:     {PROJECT_ROOT}")
    print(f"  Model path:       {model_path}")
    print(f"  Data path:        {data_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Device:           {device}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch}")
    print(f"  Image size:       {args.imgsz}")
    print(f"  Workers:          {args.workers}")
    print(f"  Resume:           {args.resume}")
    print(f"  Cache:            {args.cache}")


def train(args: argparse.Namespace) -> None:
    model_path = resolve_path(args.model, PROJECT_ROOT)
    data_path = resolve_path(args.data, PROJECT_ROOT)
    output_dir = DEFAULT_PROJECT_DIR / args.project_name
    device = choose_device(args.device)

    validate_training_inputs(args, model_path, data_path)
    print_training_summary(args, model_path, data_path, output_dir, device)

    if args.dry_run:
        print("Dry run complete. Training was not started.")
        return

    model = YOLO(str(model_path))
    model.train(
        project=str(DEFAULT_PROJECT_DIR),
        name=args.project_name,
        device=device,
        workers=args.workers,
        cache=args.cache,
        seed=0,
        agnostic_nms=False,
        verbose=True,
        plots=True,
        data=str(data_path),
        imgsz=args.imgsz,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=0.0,
        translate=0.05,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        multi_scale=False,
        close_mosaic=10,
        task="detect",
        pretrained=True,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        resume=args.resume,
        split="val",
        val=True,
        save=True,
        save_period=args.save_period,
        half=False,
        # Validation confidence threshold, not a training loss setting.
        # Service-time confidence should be tuned separately after training.
        conf=0.001,
        iou=0.7,
        optimizer="AdamW",
        cos_lr=True,
        lr0=0.001,
        lrf=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        dropout=0.0,
    )


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
