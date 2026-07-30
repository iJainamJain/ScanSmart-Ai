"""Contour extraction and document boundary detection."""

import cv2
import numpy as np

EPSILON_FRACTIONS = (0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.12)


def find_contours(binary_image: np.ndarray) -> list[np.ndarray]:
    """Find external contours in a binary image (edge map or mask)."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _score_contour(approx: np.ndarray, mask: np.ndarray, gray: np.ndarray | None, image_area: float) -> float:
    # 1. Area fraction
    area = cv2.contourArea(approx)
    area_score = area / image_area
    
    # 2. Rectangularity (contour area / bounding box area)
    rect = cv2.minAreaRect(approx)
    box_area = rect[1][0] * rect[1][1]
    rectangularity = area / box_area if box_area > 0 else 0.0
    
    # 3. Brightness and Texture
    if gray is not None:
        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [approx.astype(int)], -1, 255, -1)
        
        mean_val, std_val = cv2.meanStdDev(gray, mask=contour_mask)
        mean_val = mean_val[0][0]
        std_val = std_val[0][0]
        
        brightness_score = mean_val / 255.0
        texture_penalty = std_val / 128.0
    else:
        brightness_score = 1.0
        texture_penalty = 0.0

    # Combine scores
    score = (0.3 * area_score) + (0.3 * rectangularity) + (0.4 * brightness_score) - (0.5 * texture_penalty)
    return score


def find_document_contour(mask: np.ndarray, image_area: float, gray: np.ndarray | None = None) -> np.ndarray | None:
    """Find the document's 4 corners from a cleaned paper-segmentation mask.

    Evaluates all large candidate contours by simplifying to 4 points and scoring
    based on area, rectangularity, brightness, and texture.
    Returns the corners of the highest-scoring candidate.
    """
    contours = find_contours(mask)
    if not contours:
        return None

    best_approx = None
    best_score = -float('inf')

    # Consider contours that are at least 5% of the image area
    valid_contours = [c for c in contours if cv2.contourArea(c) > image_area * 0.05]
    
    for c in valid_contours:
        hull = cv2.convexHull(c)
        perimeter = cv2.arcLength(hull, True)
        
        for eps_fraction in EPSILON_FRACTIONS:
            approx = cv2.approxPolyDP(hull, eps_fraction * perimeter, True)
            if len(approx) == 4:
                approx_pts = approx.reshape(4, 2).astype(np.float32)
                score = _score_contour(approx_pts, mask, gray, image_area)
                if score > best_score:
                    best_score = score
                    best_approx = approx_pts
                break  # Stop epsilon loop once we find a 4-point approx for this contour

    return best_approx
