"""Thresholding techniques: paper-region segmentation and final-output binarization."""

import cv2
import numpy as np


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    """Binarize using Otsu's automatic global threshold.

    Otsu picks a threshold from the image's bimodal histogram instead of a
    fixed guess; combined with THRESH_BINARY, pixels brighter than that
    threshold are always mapped to 255 regardless of how much of the frame
    they cover, so no polarity correction is needed here.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def global_threshold(gray: np.ndarray, thresh: int = 127) -> np.ndarray:
    """Binarize using a single fixed threshold value for the whole image."""
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary


def adaptive_threshold(gray: np.ndarray, block_size: int = 25, c: int = 15) -> np.ndarray:
    """Binarize using a locally-computed threshold per pixel neighborhood.

    A single global threshold (fixed or Otsu) breaks down when lighting
    varies across the page - part of the page ends up all-black or
    all-white. Adaptive thresholding computes each pixel's threshold from
    its own neighborhood, which handles shadows/uneven lighting far better
    and is what actually produces a clean "scanned B&W page" look.
    """
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def segment_paper(gray: np.ndarray) -> np.ndarray:
    """Binarize a grayscale image so the (bright) paper is foreground (255).

    Thin wrapper around otsu_binarize - kept as a separate name because its
    purpose (finding the page's boundary in a raw photo) is distinct from
    using the same technique to binarize the final scanner output.
    """
    return otsu_binarize(gray)


def clean_mask(mask: np.ndarray, close_size: int = 5, open_size: int = 21) -> np.ndarray:
    """Morphologically clean a binary mask before contour extraction.

    Closing (small kernel) fills small holes inside the page silhouette.
    Opening (larger kernel) strips thin spurious protrusions - e.g. a stray
    bright reflection bridging into the background - that would otherwise
    drag a convex-hull boundary estimate way outside the actual page.
    """
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    return opened
