"""Visualize Pascal VOC XML annotations and save object crops."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "annotation_check"
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
COORDINATE_CLIP_TOLERANCE = 10
MIN_CROP_LONG_SIDE = 400


@dataclass(frozen=True)
class VocObject:
    """One Pascal VOC object annotation."""

    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class VocAnnotation:
    """Parsed Pascal VOC annotation data."""

    filename: str
    width: int
    height: int
    objects: list[VocObject]


def resolve_project_path(path_text: str) -> Path:
    """Resolve absolute paths and project-root-relative paths."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_required_int(element: ET.Element, tag_name: str, xml_path: Path) -> int:
    """Read a required integer XML tag."""
    text = element.findtext(tag_name)
    if text is None or not text.strip():
        raise ValueError(f"{xml_path}: missing <{tag_name}>")

    try:
        return int(round(float(text.strip())))
    except ValueError as exc:
        raise ValueError(f"{xml_path}: invalid integer in <{tag_name}>: {text!r}") from exc


def parse_voc_xml(xml_path: Path) -> VocAnnotation:
    """Parse a Pascal VOC XML file."""
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse XML: {xml_path}: {exc}") from exc

    filename = (root.findtext("filename") or xml_path.with_suffix(".jpg").name).strip()

    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path}: missing <size>")
    width = parse_required_int(size, "width", xml_path)
    height = parse_required_int(size, "height", xml_path)
    if width <= 0 or height <= 0:
        raise ValueError(f"{xml_path}: image size must be positive: {width} x {height}")

    object_elements = root.findall("object")
    if not object_elements:
        raise ValueError(f"{xml_path}: missing <object>")

    objects: list[VocObject] = []
    for object_index, obj in enumerate(object_elements, start=1):
        name = (obj.findtext("name") or "").strip()
        if not name:
            raise ValueError(f"{xml_path}: object {object_index} has no <name>")

        bbox = obj.find("bndbox")
        if bbox is None:
            raise ValueError(f"{xml_path}: object {object_index} ({name}) has no <bndbox>")

        objects.append(
            VocObject(
                name=name,
                xmin=parse_required_int(bbox, "xmin", xml_path),
                ymin=parse_required_int(bbox, "ymin", xml_path),
                xmax=parse_required_int(bbox, "xmax", xml_path),
                ymax=parse_required_int(bbox, "ymax", xml_path),
            )
        )

    return VocAnnotation(filename=filename, width=width, height=height, objects=objects)


def validate_box(
    box: VocObject,
    image_width: int,
    image_height: int,
    object_index: int,
    *,
    clip_tolerance: int = COORDINATE_CLIP_TOLERANCE,
) -> tuple[VocObject, list[str]]:
    """Validate and optionally clip one VOC bounding box."""
    if box.xmin >= box.xmax or box.ymin >= box.ymax:
        raise ValueError(
            f"object {object_index} ({box.name}) has invalid box: "
            f"xmin >= xmax or ymin >= ymax ({box.xmin}, {box.ymin}, {box.xmax}, {box.ymax})"
        )

    min_allowed = -clip_tolerance
    max_x_allowed = image_width + clip_tolerance
    max_y_allowed = image_height + clip_tolerance
    too_far_outside = (
        box.xmin < min_allowed
        or box.ymin < min_allowed
        or box.xmax > max_x_allowed
        or box.ymax > max_y_allowed
    )
    if too_far_outside:
        raise ValueError(
            f"object {object_index} ({box.name}) coordinates are outside image range: "
            f"({box.xmin}, {box.ymin}, {box.xmax}, {box.ymax}) for "
            f"{image_width} x {image_height}"
        )

    clipped = VocObject(
        name=box.name,
        xmin=max(0, min(image_width - 1, box.xmin)),
        ymin=max(0, min(image_height - 1, box.ymin)),
        xmax=max(0, min(image_width, box.xmax)),
        ymax=max(0, min(image_height, box.ymax)),
    )
    if clipped.xmin >= clipped.xmax or clipped.ymin >= clipped.ymax:
        raise ValueError(
            f"object {object_index} ({box.name}) became invalid after clipping: "
            f"({clipped.xmin}, {clipped.ymin}, {clipped.xmax}, {clipped.ymax})"
        )

    warnings: list[str] = []
    if clipped != box:
        warnings.append(
            f"[WARNING] object {object_index} ({box.name}) box clipped to image bounds: "
            f"({box.xmin}, {box.ymin}, {box.xmax}, {box.ymax}) -> "
            f"({clipped.xmin}, {clipped.ymin}, {clipped.xmax}, {clipped.ymax})"
        )

    return clipped, warnings


def color_for_name(name: str) -> tuple[int, int, int]:
    """Create a stable high-contrast BGR color for a class name."""
    palette = (
        (0, 255, 255),
        (0, 180, 255),
        (255, 120, 0),
        (80, 220, 80),
        (255, 80, 180),
        (180, 120, 255),
    )
    return palette[sum(name.encode("utf-8")) % len(palette)]


def draw_annotations(image: np.ndarray, objects: list[VocObject]) -> np.ndarray:
    """Draw all bounding boxes and object labels on an image copy."""
    annotated = image.copy()
    image_height, image_width = annotated.shape[:2]
    thickness = max(3, round(max(image_width, image_height) / 900))
    font_scale = max(0.65, max(image_width, image_height) / 2200)
    text_thickness = max(2, thickness - 1)

    for object_index, obj in enumerate(objects, start=1):
        color = color_for_name(obj.name)
        label = f"{obj.name}_{object_index}"
        top_left = (obj.xmin, obj.ymin)
        bottom_right = (obj.xmax, obj.ymax)

        cv2.rectangle(annotated, top_left, bottom_right, color, thickness)

        box_width = obj.xmax - obj.xmin
        box_height = obj.ymax - obj.ymin
        if box_width < 40 or box_height < 40:
            outer_top_left = (max(0, obj.xmin - 6), max(0, obj.ymin - 6))
            outer_bottom_right = (min(image_width - 1, obj.xmax + 6), min(image_height - 1, obj.ymax + 6))
            cv2.rectangle(annotated, outer_top_left, outer_bottom_right, color, max(2, thickness - 1))

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )
        text_width, text_height = text_size
        label_x = max(0, min(obj.xmin, image_width - text_width - 8))
        label_y = obj.ymin - 8
        if label_y - text_height - baseline < 0:
            label_y = min(image_height - 4, obj.ymax + text_height + baseline + 8)

        background_top_left = (label_x, max(0, label_y - text_height - baseline - 4))
        background_bottom_right = (
            min(image_width - 1, label_x + text_width + 8),
            min(image_height - 1, label_y + baseline + 4),
        )
        cv2.rectangle(annotated, background_top_left, background_bottom_right, color, -1)
        cv2.putText(
            annotated,
            label,
            (label_x + 4, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )

    return annotated


def safe_class_name(class_name: str) -> str:
    """Make a class name safe for filenames."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", class_name.strip())
    return safe_name.strip("._") or "object"


def save_image(path: Path, image: np.ndarray) -> None:
    """Save an image and fail loudly if OpenCV cannot write it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")


def save_object_crops(
    image: np.ndarray,
    objects: list[VocObject],
    output_dir: Path,
    padding: int,
) -> list[Path]:
    """Save padded object crop images with the local bounding box redrawn."""
    if padding < 0:
        raise ValueError(f"padding must be non-negative: {padding}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_height, image_width = image.shape[:2]
    saved_paths: list[Path] = []

    for object_index, obj in enumerate(objects, start=1):
        crop_xmin = max(0, obj.xmin - padding)
        crop_ymin = max(0, obj.ymin - padding)
        crop_xmax = min(image_width, obj.xmax + padding)
        crop_ymax = min(image_height, obj.ymax + padding)
        crop = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy()
        if crop.size == 0:
            raise ValueError(f"object {object_index} ({obj.name}) produced an empty crop")

        local_object = VocObject(
            name=obj.name,
            xmin=obj.xmin - crop_xmin,
            ymin=obj.ymin - crop_ymin,
            xmax=obj.xmax - crop_xmin,
            ymax=obj.ymax - crop_ymin,
        )
        crop = draw_annotations(crop, [local_object])

        crop_height, crop_width = crop.shape[:2]
        long_side = max(crop_width, crop_height)
        if long_side < MIN_CROP_LONG_SIDE:
            scale = MIN_CROP_LONG_SIDE / long_side
            resized_width = max(1, round(crop_width * scale))
            resized_height = max(1, round(crop_height * scale))
            crop = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

        output_path = output_dir / f"object_{object_index:02d}_{safe_class_name(obj.name)}.jpg"
        save_image(output_path, crop)
        saved_paths.append(output_path)

    return saved_paths


def process_single_pair(image_path: Path, xml_path: Path, padding: int) -> bool:
    """Process one image/XML pair."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    annotation = parse_voc_xml(xml_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image_height, image_width = image.shape[:2]
    if (annotation.width, annotation.height) != (image_width, image_height):
        raise ValueError(
            f"Image and XML size mismatch for {image_path.name}: "
            f"image={image_width} x {image_height}, "
            f"xml={annotation.width} x {annotation.height}"
        )

    validated_objects: list[VocObject] = []
    warnings: list[str] = []
    for object_index, obj in enumerate(annotation.objects, start=1):
        validated_object, object_warnings = validate_box(
            obj,
            image_width,
            image_height,
            object_index,
        )
        validated_objects.append(validated_object)
        warnings.extend(object_warnings)

    boxed_image = draw_annotations(image, validated_objects)
    boxed_output_path = OUTPUT_ROOT / f"{image_path.stem}_boxed.jpg"
    save_image(boxed_output_path, boxed_image)

    crop_output_dir = OUTPUT_ROOT / image_path.stem
    crop_paths = save_object_crops(image, validated_objects, crop_output_dir, padding)

    class_counts = Counter(obj.name for obj in validated_objects)
    for warning in warnings:
        print(warning)
    print(f"[OK] {image_path.name}")
    print(f"- image size: {image_width} x {image_height}")
    print(f"- objects: {len(validated_objects)}")
    for class_name, count in sorted(class_counts.items()):
        print(f"- {class_name}: {count}")
    print(f"- saved boxed image: {boxed_output_path}")
    print(f"- saved crops: {len(crop_paths)}")
    return True


def find_image_for_stem(image_dir: Path, stem: str) -> Path | None:
    """Find the first supported image path matching a stem."""
    for extension in SUPPORTED_IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
        uppercase_candidate = image_dir / f"{stem}{extension.upper()}"
        if uppercase_candidate.exists():
            return uppercase_candidate
    return None


def collect_image_stems(image_dir: Path) -> set[str]:
    """Collect supported image stems from a directory."""
    stems: set[str] = set()
    for extension in SUPPORTED_IMAGE_EXTENSIONS:
        stems.update(path.stem for path in image_dir.glob(f"*{extension}"))
        stems.update(path.stem for path in image_dir.glob(f"*{extension.upper()}"))
    return stems


def process_directory(image_dir: Path, xml_dir: Path, padding: int) -> int:
    """Process all matching image/XML pairs in two directories."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not xml_dir.exists():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")

    image_stems = collect_image_stems(image_dir)
    xml_stems = {path.stem for path in xml_dir.glob("*.xml")}
    all_stems = sorted(image_stems | xml_stems)
    if not all_stems:
        raise ValueError(f"No supported images or XML files found in: {image_dir}, {xml_dir}")

    success_count = 0
    failure_count = 0
    for stem in all_stems:
        image_path = find_image_for_stem(image_dir, stem)
        xml_path = xml_dir / f"{stem}.xml"

        try:
            if image_path is None:
                raise FileNotFoundError(f"Image file not found for XML stem '{stem}' in: {image_dir}")
            if not xml_path.exists():
                raise FileNotFoundError(f"XML file not found for image stem '{stem}' in: {xml_dir}")
            process_single_pair(image_path, xml_path, padding)
            success_count += 1
        except Exception as exc:
            failure_count += 1
            print(f"[ERROR] {stem}: {exc}", file=sys.stderr)

    print("[SUMMARY]")
    print(f"- processed: {success_count}")
    print(f"- failed: {failure_count}")
    print(f"- output dir: {OUTPUT_ROOT}")
    return failure_count


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Pascal VOC XML bounding boxes and save padded object crops."
        )
    )
    parser.add_argument("--image", help="Path to one source image.")
    parser.add_argument("--xml", help="Path to one Pascal VOC XML annotation.")
    parser.add_argument("--image-dir", help="Directory containing source images.")
    parser.add_argument("--xml-dir", help="Directory containing Pascal VOC XML files.")
    parser.add_argument(
        "--padding",
        type=int,
        default=80,
        help="Padding in pixels around each object crop. Default: 80.",
    )
    return parser


def main() -> int:
    """Run the VOC visualization command-line tool."""
    parser = build_parser()
    args = parser.parse_args()

    single_mode = args.image is not None or args.xml is not None
    directory_mode = args.image_dir is not None or args.xml_dir is not None
    if single_mode == directory_mode:
        parser.error("Use either --image/--xml or --image-dir/--xml-dir.")
    if single_mode and (args.image is None or args.xml is None):
        parser.error("Single-file mode requires both --image and --xml.")
    if directory_mode and (args.image_dir is None or args.xml_dir is None):
        parser.error("Directory mode requires both --image-dir and --xml-dir.")
    if args.padding < 0:
        parser.error("--padding must be non-negative.")

    try:
        if single_mode:
            process_single_pair(
                resolve_project_path(args.image),
                resolve_project_path(args.xml),
                args.padding,
            )
            return 0

        failure_count = process_directory(
            resolve_project_path(args.image_dir),
            resolve_project_path(args.xml_dir),
            args.padding,
        )
        return 1 if failure_count else 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
