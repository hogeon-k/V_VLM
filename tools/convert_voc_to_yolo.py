"""Convert Pascal VOC XML labels to YOLO TXT labels."""

from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Change these paths to match your dataset layout.
XML_DIR = PROJECT_ROOT / "data" / "annotations"
LABELS_DIR = PROJECT_ROOT / "labels"

CLASS_MAP = {
    "open_circuit": 0,
    "short": 1,
    "missing_hole": 2,
}


def voc_bbox_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    """Convert Pascal VOC bbox coordinates to normalized YOLO coordinates."""
    x_center = ((xmin + xmax) / 2) / image_width
    y_center = ((ymin + ymax) / 2) / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    return x_center, y_center, width, height


def parse_size(root: ET.Element, xml_path: Path) -> tuple[float, float]:
    """Read image width and height from a Pascal VOC XML root."""
    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path}: missing <size> element")

    width_text = size.findtext("width")
    height_text = size.findtext("height")
    if width_text is None or height_text is None:
        raise ValueError(f"{xml_path}: missing image width or height")

    image_width = float(width_text)
    image_height = float(height_text)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"{xml_path}: image width and height must be positive")

    return image_width, image_height


def convert_xml_file(xml_path: Path, output_path: Path) -> tuple[int, int]:
    """Convert one XML file and return written and skipped object counts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_width, image_height = parse_size(root, xml_path)

    yolo_rows: list[str] = []
    skipped_count = 0
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        if class_name not in CLASS_MAP:
            print(f"Warning: {xml_path.name}: unknown class '{class_name}', skipped")
            skipped_count += 1
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            print(f"Warning: {xml_path.name}: object '{class_name}' has no bndbox, skipped")
            skipped_count += 1
            continue

        try:
            xmin = float(bbox.findtext("xmin", ""))
            ymin = float(bbox.findtext("ymin", ""))
            xmax = float(bbox.findtext("xmax", ""))
            ymax = float(bbox.findtext("ymax", ""))
        except ValueError:
            print(f"Warning: {xml_path.name}: invalid bbox for '{class_name}', skipped")
            skipped_count += 1
            continue

        x_center, y_center, width, height = voc_bbox_to_yolo(
            xmin,
            ymin,
            xmax,
            ymax,
            image_width,
            image_height,
        )

        class_id = CLASS_MAP[class_name]
        # YOLO expects one object per line: class_id and normalized bbox values.
        yolo_rows.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    output_path.write_text("\n".join(yolo_rows) + ("\n" if yolo_rows else ""), encoding="utf-8")
    return len(yolo_rows), skipped_count


def main() -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    if not XML_DIR.exists():
        raise FileNotFoundError(f"XML folder does not exist: {XML_DIR}")

    xml_count = 0
    txt_count = 0
    skipped_object_count = 0
    for xml_path in sorted(XML_DIR.glob("*.xml")):
        output_path = LABELS_DIR / f"{xml_path.stem}.txt"
        _, skipped_count = convert_xml_file(xml_path, output_path)
        skipped_object_count += skipped_count
        xml_count += 1
        txt_count += 1

    print(f"Total XML files: {xml_count}")
    print(f"Created TXT files: {txt_count}")
    print(f"Skipped objects: {skipped_object_count}")
    print(f"YOLO labels saved to: {LABELS_DIR}")


if __name__ == "__main__":
    main()
