"""Refine a detected document quadrilateral by snapping its edges to the real page border.

Otsu segmentation merges the page with any bright background it touches (a
patterned cloth, a light table), so the detected quadrilateral is often a
superset of the page. Global fixes don't work here: page and background
overlap in both brightness and local variance, and the usable threshold
shifts per photo with exposure. This module instead calibrates "what this
page looks like" from the interior of the detected quad, then walks each
edge inward until those statistics start holding.
"""

import cv2
import numpy as np

from src.perspective.transform import order_points

# A refinement that removes more than this fraction of the detected area is
# cutting into page content rather than trimming background, so it's rejected.
MIN_AREA_RATIO = 0.75

SAMPLES_PER_EDGE = 24
MIN_INLIERS = 6
SUSTAINED_RUN_PX = 12


def _local_std(gray: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Per-pixel local standard deviation, via E[x^2] - E[x]^2 over a box filter."""
    f = gray.astype(np.float32)
    mean = cv2.blur(f, (kernel_size, kernel_size))
    mean_sq = cv2.blur(f * f, (kernel_size, kernel_size))
    return np.sqrt(np.clip(mean_sq - mean * mean, 0, None))


def _pagelike_map(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Boolean map of pixels matching the page's own brightness/smoothness.

    The reference is sampled from the middle of the detected quad, which is
    page even when the edges have over-reached, making this robust to the
    exposure differences that defeat any fixed global threshold.
    """
    ordered = order_points(quad)
    centre = ordered.mean(axis=0)
    inner = (centre + (ordered - centre) * 0.45).astype(np.int32)

    interior = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(interior, [inner], 255)

    std = _local_std(gray)
    reference_mean = float(np.mean(gray[interior > 0]))
    reference_std = float(np.percentile(std[interior > 0], 85))
    return (gray > reference_mean - 35) & (std < reference_std + 12)


def _intersect(line_a, line_b):
    (a0, a1), (b0, b1) = line_a, line_b
    da, db = a1 - a0, b1 - b0
    denominator = da[0] * db[1] - da[1] * db[0]
    if abs(denominator) < 1e-9:
        return None
    t = ((b0[0] - a0[0]) * db[1] - (b0[1] - a0[1]) * db[0]) / denominator
    return a0 + t * da


def refine_quad(quad: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Snap each edge of `quad` inward onto the page border; returns the original on failure.

    Each edge is sampled at several points, each sample marched inward until
    it enters a sustained run of page-like pixels, and a line re-fitted to
    those hits - fitting rather than translating so a mis-angled edge is
    corrected too. Adjacent fitted lines are intersected for the new corners.
    """
    ordered = order_points(quad).astype(np.float64)
    centre = ordered.mean(axis=0)
    pagelike = _pagelike_map(gray, quad)
    height, width = gray.shape

    fitted_lines = []
    for i in range(4):
        start, end = ordered[i], ordered[(i + 1) % 4]
        midpoint = start + 0.5 * (end - start)
        inward = centre - midpoint
        distance_to_centre = np.linalg.norm(inward)
        if distance_to_centre < 1e-6:
            return quad
        inward = inward / distance_to_centre
        search_limit = int(distance_to_centre * 0.55)

        hits = []
        for t in np.linspace(0.12, 0.88, SAMPLES_PER_EDGE):
            origin = start + t * (end - start)
            for step in range(0, search_limit, 2):
                probe = origin + inward * step
                if _sustained_pagelike(pagelike, probe, inward, width, height):
                    hits.append(probe)
                    break

        if len(hits) < MIN_INLIERS:
            fitted_lines.append((start, end))
            continue

        vx, vy, x0, y0 = cv2.fitLine(
            np.array(hits, np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        ).flatten()
        fitted_lines.append((np.array([x0, y0]), np.array([x0 + vx, y0 + vy])))

    corners = []
    for i in range(4):
        corner = _intersect(fitted_lines[i], fitted_lines[(i + 1) % 4])
        if corner is None:
            return quad
        corners.append(corner)

    # _intersect(edge_i, edge_i+1) yields the corner *after* edge i, so the
    # result starts one position ahead of the input ordering.
    refined = np.roll(np.array(corners, np.float32), 1, axis=0)
    if not np.all(np.isfinite(refined)):
        return quad

    original_area = cv2.contourArea(order_points(quad).astype(np.float32))
    refined_area = cv2.contourArea(order_points(refined).astype(np.float32))
    if original_area <= 0:
        return quad
    ratio = refined_area / original_area
    if ratio < MIN_AREA_RATIO or ratio > 1.05:
        return quad
    return refined


def _sustained_pagelike(pagelike, probe, inward, width, height) -> bool:
    """True if the page-like condition holds for a short run ahead of `probe`.

    Requiring a run rather than a single pixel stops the search from halting
    on one bright square of a patterned background.
    """
    for offset in range(0, SUSTAINED_RUN_PX, 3):
        point = probe + inward * offset
        x, y = int(round(point[0])), int(round(point[1]))
        if not (0 <= x < width and 0 <= y < height) or not pagelike[y, x]:
            return False
    return True
