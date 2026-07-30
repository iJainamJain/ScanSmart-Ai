"""Image loading and resizing utilities."""

from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk in BGR color order.

    Raises FileNotFoundError if the path doesn't exist or OpenCV can't decode it.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not decode image: {image_path}")

    return image


def resize_image(image: np.ndarray, max_dimension: int = 1500) -> tuple[np.ndarray, float]:
    """Resize so the longer side is at most max_dimension, preserving aspect ratio.

    Large input photos slow down contour/edge detection without adding useful
    detail, so downscaling first speeds up the whole pipeline.
    Returns (resized_image, scale_factor) so callers can map coordinates
    back to the original image if needed.
    """
    height, width = image.shape[:2]
    longer_side = max(height, width)

    if longer_side <= max_dimension:
        return image.copy(), 1.0

    scale = max_dimension / longer_side
    new_size = (int(width * scale), int(height * scale))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale
