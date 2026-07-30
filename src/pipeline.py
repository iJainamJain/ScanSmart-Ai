"""Shared pipeline steps used by the CLI, the GUI, and the evaluator.

These were duplicated across main.py, app/main.py and the evaluation module,
which meant a tuning change in one place silently diverged from the others.
"""

import numpy as np

from src.detection.contours import find_document_contour
from src.detection.edges import denoise, to_grayscale
from src.detection.refine import refine_quad
from src.enhancement.basic import enhance_document
from src.enhancement.contrast import adjust_brightness_contrast
from src.enhancement.sharpen import sharpen
from src.morphology.operations import closing, opening
from src.perspective.transform import four_point_transform
from src.segmentation.threshold import adaptive_threshold, clean_mask, segment_paper

BRIGHTNESS = 10
CONTRAST = 1.15
MORPH_KERNEL = 3


def detect_document(resized: np.ndarray) -> np.ndarray | None:
    """Find the page's 4 corners in a resized photo, refined onto the real border."""
    gray = to_grayscale(resized)
    mask = clean_mask(segment_paper(denoise(gray)))
    image_area = resized.shape[0] * resized.shape[1]
    corners = find_document_contour(mask, image_area, gray)
    if corners is None:
        return None
    return refine_quad(corners, gray)


def enhance(flattened: np.ndarray) -> np.ndarray:
    """Brightness/contrast, sharpening, then CLAHE - the greyscale 'scanned' look."""
    return enhance_document(sharpen(adjust_brightness_contrast(flattened, BRIGHTNESS, CONTRAST)))


def binarize(enhanced: np.ndarray) -> np.ndarray:
    """Adaptive threshold plus morphological cleanup - the final black & white page."""
    return closing(opening(adaptive_threshold(enhanced), MORPH_KERNEL), MORPH_KERNEL)


def scan_page(resized: np.ndarray, corners: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (if corners are known), enhance and binarize. Returns (enhanced, final_bw)."""
    flattened = four_point_transform(resized, corners) if corners is not None else resized
    enhanced = enhance(flattened)
    return enhanced, binarize(enhanced)
