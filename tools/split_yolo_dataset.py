"""Split YOLO image/label pairs into train, val, and test datasets."""

from pathlib import Path
import random
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Change these paths to match your dataset layout.
IMAGE_DIR = "data/images"
LABEL_DIR = "labels"
OUTPUT_DIR = "datasets/pcb"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
CLASS_NAMES = [
    "open_circuit",
    "short",
    "missing_hole",
]


def project_path(path_text: str) -> Path:
    """Resolve a top-level config path from the project root."""
    return PROJECT_ROOT / path_text


def collect_image_label_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    """Collect images that have a YOLO txt label with the same stem."""
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"Warning: {image_path.name}: matching label file not found, skipped")
            continue

        pairs.append((image_path, label_path))
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
) -> dict[str, list[tuple[Path, Path]]]:
    """Shuffle and split image/label pairs by the configured ratios."""
    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise ValueError("TRAIN_RATIO, VAL_RATIO, and TEST_RATIO must add up to 1.0")

    shuffled_pairs = pairs[:]
    random.Random(RANDOM_SEED).shuffle(shuffled_pairs)

    total_count = len(shuffled_pairs)
    raw_counts = {
        "train": total_count * TRAIN_RATIO,
        "val": total_count * VAL_RATIO,
        "test": total_count * TEST_RATIO,
    }
    split_counts = {split_name: int(count) for split_name, count in raw_counts.items()}

    # Distribute leftover items to the splits with the largest decimal remainders.
    remaining_count = total_count - sum(split_counts.values())
    sorted_remainders = sorted(
        SPLITS,
        key=lambda split_name: (raw_counts[split_name] - split_counts[split_name]),
        reverse=True,
    )
    for split_name in sorted_remainders[:remaining_count]:
        split_counts[split_name] += 1

    train_end = split_counts["train"]
    val_end = train_end + split_counts["val"]

    return {
        "train": shuffled_pairs[:train_end],
        "val": shuffled_pairs[train_end:val_end],
        "test": shuffled_pairs[val_end:],
    }


def prepare_output_dirs(output_dir: Path) -> None:
    """Create YOLO dataset image and label folders for each split."""
    for split_name in SPLITS:
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)


def copy_split_files(
    split_name: str,
    pairs: list[tuple[Path, Path]],
    output_dir: Path,
) -> None:
    """Copy image files and their matching label files into one split."""
    image_output_dir = output_dir / "images" / split_name
    label_output_dir = output_dir / "labels" / split_name

    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_output_dir / image_path.name)
        shutil.copy2(label_path, label_output_dir / label_path.name)


def write_data_yaml(output_dir: Path) -> None:
    """Write a YOLO data.yaml file for the PCB defect classes."""
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    data_yaml = (
        f"path: {OUTPUT_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n"
        f"{names}\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def main() -> None:
    image_dir = project_path(IMAGE_DIR)
    label_dir = project_path(LABEL_DIR)
    output_dir = project_path(OUTPUT_DIR)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label folder does not exist: {label_dir}")

    pairs = collect_image_label_pairs(image_dir, label_dir)
    split_map = split_pairs(pairs)

    prepare_output_dirs(output_dir)
    for split_name, split_items in split_map.items():
        copy_split_files(split_name, split_items, output_dir)

    write_data_yaml(output_dir)

    print(f"Total image/label pairs: {len(pairs)}")
    print(f"Train: {len(split_map['train'])}")
    print(f"Val: {len(split_map['val'])}")
    print(f"Test: {len(split_map['test'])}")
    print(f"YOLO dataset saved to: {output_dir}")
    print(f"data.yaml saved to: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
