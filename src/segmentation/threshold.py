"""Otsu thresholding to segment the document page from its background."""

import cv2
import numpy as np


def segment_paper(gray: np.ndarray) -> np.ndarray:
    """Binarize a grayscale image so the (bright) paper is foreground (255).

    Otsu picks a global threshold automatically from the image's bimodal
    histogram (paper vs. background). Since Otsu doesn't know which side is
    "paper", we flip polarity when the foreground class turns out to be the
    majority of the frame - the page is usually the brighter, smaller region
    against a larger/darker surroundings.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        return binary
    return cv2.bitwise_not(binary)


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
