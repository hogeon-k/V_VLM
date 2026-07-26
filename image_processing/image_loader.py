from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image_unicode(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise OSError(f"Failed to read image file: {path}; {exc}") from exc

    if raw.size == 0:
        raise ValueError(f"Input image file is empty: {path}")

    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Input image is not decodable: {path}")
    if image.dtype != np.uint8:
        raise ValueError(f"Decoded image must be uint8: {path}; dtype={image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Decoded image must be HWC BGR with 3 channels: {path}; shape={image.shape}")
    return image


class ImageLoader:
    def load(self, image_path: str | Path) -> np.ndarray:
        return read_image_unicode(image_path)
