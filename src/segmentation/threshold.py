"""Otsu thresholding to segment the document page from its background."""

import cv2
import numpy as np


def segment_paper(gray: np.ndarray) -> np.ndarray:
    """Binarize a grayscale image so the (bright) paper is foreground (255).

    Otsu picks a threshold from the image's bimodal histogram; combined
    with THRESH_BINARY, pixels brighter than that threshold are always
    mapped to 255 regardless of how much of the frame they cover, so no
    polarity correction is needed here.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


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
