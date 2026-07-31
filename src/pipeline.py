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
from src.enhancement.illumination import flatten_illumination
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


def enhance(page: np.ndarray, illumination_normalized: bool = False) -> np.ndarray:
    """Brightness/contrast, sharpening, then CLAHE - the greyscale 'scanned' look.

    Set `illumination_normalized` when the page has already been through
    illumination flattening. That step leaves the paper at ~255, so applying
    the brightness/contrast lift on top saturates the page and crushes the
    ink-to-paper contrast the thresholding depends on: measured across six
    real photos, ink coverage in the final output collapsed from 10.4% to
    1.4%, erasing most of the handwriting. Skipping the lift restores it to
    10.6%. Sharpening and CLAHE remain safe and are still applied.
    """
    stage = page if illumination_normalized else adjust_brightness_contrast(page, BRIGHTNESS, CONTRAST)
    return enhance_document(sharpen(stage))


def binarize(enhanced: np.ndarray) -> np.ndarray:
    """Adaptive threshold plus morphological cleanup - the final black & white page."""
    return closing(opening(adaptive_threshold(enhanced), MORPH_KERNEL), MORPH_KERNEL)


def scan_page(
    resized: np.ndarray,
    corners: np.ndarray | None,
    flatten_lighting: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (if corners are known), enhance and binarize. Returns (enhanced, final_bw).

    `flatten_lighting` additionally removes the illumination field, which
    rescues heavily shadowed or unevenly lit pages but is subtractive - see
    src/enhancement/illumination.py before enabling it by default.
    """
    page = four_point_transform(resized, corners) if corners is not None else resized

    if flatten_lighting:
        gray = to_grayscale(page) if page.ndim == 3 else page
        page = flatten_illumination(gray)

    enhanced = enhance(page, illumination_normalized=flatten_lighting)
    return enhanced, binarize(enhanced)
