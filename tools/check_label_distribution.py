"""Check YOLO label class distribution for train, val, and test splits."""

from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS_ROOT = PROJECT_ROOT / "datasets" / "pcb" / "labels"
SPLITS = ("train", "val", "test")
CLASS_NAMES = {
    0: "open_circuit",
    1: "short",
    2: "missing_hole",
}


def count_labels(label_dir: Path, split_name: str) -> Counter[int]:
    """Count class IDs from YOLO txt labels in one split folder."""
    counts: Counter[int] = Counter()

    if not label_dir.exists():
        print(f"INFO: {split_name} label folder does not exist: {label_dir}")
        return counts

    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        print(f"INFO: {split_name} split has no label txt files: {label_dir}")
        return counts

    for label_file in label_files:
        try:
            lines = label_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            print(f"WARNING: could not read {label_file}: {exc}")
            continue

        for line_number, line in enumerate(lines, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # YOLO label format starts with class_id.
            class_id_text = stripped_line.split()[0]
            try:
                class_id = int(class_id_text)
            except ValueError:
                print(
                    f"WARNING: {label_file.name}:{line_number} has invalid class_id "
                    f"'{class_id_text}', skipped"
                )
                continue

            if class_id not in CLASS_NAMES:
                print(
                    f"WARNING: {label_file.name}:{line_number} has unknown class_id "
                    f"{class_id}, skipped"
                )
                continue

            counts[class_id] += 1

    return counts


def print_counts(split_name: str, counts: Counter[int]) -> None:
    """Print class counts and warn when a class has no labels."""
    print(f"[{split_name}]")
    for class_id, class_name in CLASS_NAMES.items():
        count = counts[class_id]
        print(f"{class_id} {class_name}: {count:02d}")

    for class_id, class_name in CLASS_NAMES.items():
        if counts[class_id] == 0:
            print(
                f"WARNING: {split_name} split has 0 labels for class "
                f"{class_id} {class_name}"
            )
    print()


def main() -> None:
    total_counts: Counter[int] = Counter()

    for split_name in SPLITS:
        split_counts = count_labels(LABELS_ROOT / split_name, split_name)
        total_counts.update(split_counts)
        print_counts(split_name, split_counts)

    print_counts("total", total_counts)


if __name__ == "__main__":
    main()
