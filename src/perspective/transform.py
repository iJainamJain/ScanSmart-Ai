"""Corner ordering and perspective (four-point) transform."""

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left.

    cv2.approxPolyDP returns corners in no particular order, but a
    perspective transform needs a consistent order to map them correctly.
    The top-left point has the smallest x+y sum and the bottom-right the
    largest; the top-right has the smallest x-y difference and the
    bottom-left the largest.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]

    diffs = np.diff(points, axis=1).flatten()
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]

    return ordered


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Warp the quadrilateral region defined by points into a flat, top-down image."""
    top_left, top_right, bottom_right, bottom_left = order_points(points)

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    max_height = int(max(height_left, height_right))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    source = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))
