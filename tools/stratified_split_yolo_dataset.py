"""Rebuild YOLO train/val/test splits with class-aware stratification."""

from collections import Counter, defaultdict
from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "pcb"
SOURCE_IMAGE_DIR = PROJECT_ROOT / "data" / "images"
SOURCE_LABEL_DIR = PROJECT_ROOT / "labels"
IMAGES_ROOT = DATASET_ROOT / "images"
LABELS_ROOT = DATASET_ROOT / "labels"

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

CLASS_NAMES = {
    0: "open_circuit",
    1: "short",
    2: "missing_hole",
}


def read_label_class_counts(label_path: Path) -> Counter[int] | None:
    """Read a YOLO label file and count valid class IDs."""
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"WARNING: could not read label file {label_path}: {exc}")
        return None

    counts: Counter[int] = Counter()
    for line_number, line in enumerate(lines, start=1):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        class_id_text = stripped_line.split()[0]
        try:
            class_id = int(class_id_text)
        except ValueError:
            print(
                f"WARNING: {label_path.name}:{line_number} has invalid class_id "
                f"'{class_id_text}', skipped"
            )
            continue

        if class_id not in CLASS_NAMES:
            print(
                f"WARNING: {label_path.name}:{line_number} has unknown class_id "
                f"{class_id}, skipped"
            )
            continue

        counts[class_id] += 1

    if not counts:
        print(f"WARNING: empty label file or no valid labels, skipped: {label_path}")
        return None

    return counts


def representative_class_id(class_counts: Counter[int]) -> int:
    """Choose the most frequent class ID, using the smallest class ID on ties."""
    max_count = max(class_counts.values())
    candidates = [
        class_id for class_id, count in class_counts.items() if count == max_count
    ]
    return min(candidates)


def collect_image_label_pairs() -> list[tuple[str, bytes, str, bytes, int]]:
    """Collect source image/label pairs before rebuilding output folders."""
    pairs: list[tuple[str, bytes, str, bytes, int]] = []
    seen_stems: set[str] = set()

    missing_roots = [
        path for path in (SOURCE_IMAGE_DIR, SOURCE_LABEL_DIR) if not path.exists()
    ]
    if missing_roots:
        print("ERROR: required source folders are missing.")
        for missing_root in missing_roots:
            print(f"- {missing_root}")
        return pairs

    for image_path in sorted(SOURCE_IMAGE_DIR.iterdir()):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            continue

        if image_path.stem in seen_stems:
            print(
                f"WARNING: duplicate image stem '{image_path.stem}', skipped: {image_path}"
            )
            continue

        label_path = SOURCE_LABEL_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"WARNING: label not found for image, skipped: {image_path}")
            continue

        class_counts = read_label_class_counts(label_path)
        if class_counts is None:
            continue

        try:
            image_bytes = image_path.read_bytes()
            label_bytes = label_path.read_bytes()
        except OSError as exc:
            print(f"WARNING: could not load pair into memory, skipped: {exc}")
            continue

        pairs.append(
            (
                image_path.name,
                image_bytes,
                label_path.name,
                label_bytes,
                representative_class_id(class_counts),
            )
        )
        seen_stems.add(image_path.stem)

    return pairs


def split_group(
    items: list[tuple[str, bytes, str, bytes, int]],
) -> dict[str, list[tuple[str, bytes, str, bytes, int]]]:
    """Split one representative-class group into train, val, and test."""
    shuffled_items = items[:]
    random.Random(RANDOM_SEED).shuffle(shuffled_items)

    total_count = len(shuffled_items)
    raw_counts = {
        "train": total_count * TRAIN_RATIO,
        "val": total_count * VAL_RATIO,
        "test": total_count * TEST_RATIO,
    }
    split_counts = {split_name: int(count) for split_name, count in raw_counts.items()}

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
        "train": shuffled_items[:train_end],
        "val": shuffled_items[train_end:val_end],
        "test": shuffled_items[val_end:],
    }


def stratified_split(
    pairs: list[tuple[str, bytes, str, bytes, int]],
) -> dict[str, list[tuple[str, bytes, str, bytes, int]]]:
    """Group by representative class and split each group independently."""
    grouped_pairs: dict[int, list[tuple[str, bytes, str, bytes, int]]] = defaultdict(
        list
    )
    for item in pairs:
        grouped_pairs[item[4]].append(item)

    split_map: dict[str, list[tuple[str, bytes, str, bytes, int]]] = {
        split_name: [] for split_name in SPLITS
    }
    for class_id in sorted(grouped_pairs):
        group_split = split_group(grouped_pairs[class_id])
        for split_name in SPLITS:
            split_map[split_name].extend(group_split[split_name])

    for split_name in SPLITS:
        random.Random(RANDOM_SEED).shuffle(split_map[split_name])

    return split_map


def rebuild_split_dirs() -> None:
    """Delete old split folders and recreate an empty YOLO split structure."""
    for split_name in SPLITS:
        for root in (IMAGES_ROOT, LABELS_ROOT):
            split_dir = root / split_name
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)

    for cache_file in LABELS_ROOT.glob("*.cache"):
        try:
            cache_file.unlink()
        except OSError as exc:
            print(f"WARNING: could not remove cache file {cache_file}: {exc}")


def copy_split_files(
    split_map: dict[str, list[tuple[str, bytes, str, bytes, int]]],
) -> None:
    """Copy image and label files into the rebuilt split folders."""
    for split_name, items in split_map.items():
        image_output_dir = IMAGES_ROOT / split_name
        label_output_dir = LABELS_ROOT / split_name

        for image_name, image_bytes, label_name, label_bytes, _ in items:
            (image_output_dir / image_name).write_bytes(image_bytes)
            (label_output_dir / label_name).write_bytes(label_bytes)


def write_data_yaml() -> None:
    """Write a YOLO data.yaml file for the 3-class PCB dataset."""
    names = "\n".join(
        f"  {class_id}: {class_name}" for class_id, class_name in CLASS_NAMES.items()
    )
    data_yaml = (
        "path: datasets/pcb\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names}\n"
    )
    (DATASET_ROOT / "data.yaml").write_text(data_yaml, encoding="utf-8")


def count_split_labels(split_name: str) -> Counter[int]:
    """Count all class IDs from labels in one rebuilt split."""
    counts: Counter[int] = Counter()
    label_dir = LABELS_ROOT / split_name
    if not label_dir.exists():
        print(f"INFO: {split_name} label folder does not exist: {label_dir}")
        return counts

    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        print(f"INFO: {split_name} split has no label txt files: {label_dir}")
        return counts

    for label_file in label_files:
        class_counts = read_label_class_counts(label_file)
        if class_counts is not None:
            counts.update(class_counts)
    return counts


def print_counts(split_name: str, counts: Counter[int]) -> None:
    """Print class counts and warn when a split has no labels for a class."""
    print(f"[{split_name}]")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"{class_id} {class_name}: {counts[class_id]:02d}")

    for class_id, class_name in CLASS_NAMES.items():
        if counts[class_id] == 0:
            print(
                f"WARNING: {split_name} split has 0 labels for class "
                f"{class_id} {class_name}"
            )
    print()


def main() -> None:
    pairs = collect_image_label_pairs()
    if not pairs:
        print("ERROR: no valid image/label pairs found. Dataset was not modified.")
        return

    split_map = stratified_split(pairs)
    rebuild_split_dirs()
    copy_split_files(split_map)
    write_data_yaml()

    total_counts: Counter[int] = Counter()
    for split_name in SPLITS:
        split_counts = count_split_labels(split_name)
        total_counts.update(split_counts)
        print_counts(split_name, split_counts)

    print_counts("total", total_counts)


if __name__ == "__main__":
    main()
