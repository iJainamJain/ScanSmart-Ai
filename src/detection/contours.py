"""Contour extraction and document boundary detection."""

import cv2
import numpy as np

EPSILON_FRACTIONS = (0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.12)


def find_contours(binary_image: np.ndarray) -> list[np.ndarray]:
    """Find external contours in a binary image (edge map or mask)."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_document_contour(mask: np.ndarray, image_area: float) -> np.ndarray | None:
    """Find the document's 4 corners from a cleaned paper-segmentation mask.

    The largest blob in the mask is assumed to be the page. Its raw
    contour is rarely an exact quadrilateral (curled corners, folds, small
    mask noise), so we take its convex hull first and simplify that with
    approxPolyDP at increasing epsilon until exactly 4 corners remain.
    Returns None if no sufficiently large candidate exists.
    """
    contours = find_contours(mask)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < image_area * 0.2:
        return None

    hull = cv2.convexHull(largest)
    perimeter = cv2.arcLength(hull, True)
    for eps_fraction in EPSILON_FRACTIONS:
        approx = cv2.approxPolyDP(hull, eps_fraction * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None
