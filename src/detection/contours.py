"""Contour extraction and document boundary detection."""

import cv2
import numpy as np


def find_contours(edge_image: np.ndarray) -> list[np.ndarray]:
    """Find external contours in a binary edge image."""
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_document_contour(contours: list[np.ndarray], image_area: float) -> np.ndarray | None:
    """Pick the largest 4-point contour that plausibly represents a document page.

    We only accept quadrilaterals covering a reasonable fraction of the frame;
    smaller shapes are usually background clutter, not the document itself.
    Returns None if no suitable contour is found.
    """
    min_area = image_area * 0.2

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            break

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None
