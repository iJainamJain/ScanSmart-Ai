"""Basic image enhancement applied after perspective correction."""

import cv2
import numpy as np


def enhance_document(image: np.ndarray) -> np.ndarray:
    """Improve contrast on the flattened document using CLAHE on grayscale.

    Scanned photos often have uneven lighting across the page; CLAHE adapts
    contrast locally (per tile) instead of globally, which handles that
    better than plain histogram equalization.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)
