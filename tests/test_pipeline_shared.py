import cv2
import numpy as np

from src.detection.contours import _score_contour, find_document_contour
from src.pipeline import binarize, detect_document, enhance, scan_page


def _photo(page=(100, 100, 400, 500), shape=(600, 500)):
    """Bright page on a darker background, as a 3-channel photo."""
    img = np.full((*shape, 3), 70, np.uint8)
    x0, y0, x1, y1 = page
    img[y0:y1, x0:x1] = 220
    return img


def test_detect_document_finds_the_page_corners():
    corners = detect_document(_photo())

    assert corners is not None
    assert corners.shape == (4, 2)
    assert corners[:, 0].min() >= 80 and corners[:, 0].max() <= 420


def test_detect_document_returns_a_full_frame_quad_when_there_is_no_border():
    """Known limitation, pinned deliberately. With no visible page border
    Otsu marks the whole frame as foreground, so detection returns a
    frame-sized quad instead of admitting failure. This is why the
    evaluation report's detection rate is labelled "found", not "correct",
    and why the close-up photos in dataset/raw cannot be scored
    automatically (see docs/dataset.md)."""
    corners = detect_document(np.full((400, 400, 3), 200, np.uint8))

    assert corners is not None
    width = corners[:, 0].max() - corners[:, 0].min()
    assert width > 390, "the returned quad spans essentially the whole frame"


def test_scan_page_returns_enhanced_and_binarized_outputs():
    photo = _photo()
    corners = detect_document(photo)

    enhanced, final_bw = scan_page(photo, corners)

    assert enhanced.ndim == 2, "enhanced output is greyscale"
    assert set(np.unique(final_bw)).issubset({0, 255}), "final output is strictly binary"


def test_scan_page_without_corners_falls_back_to_the_whole_image():
    photo = _photo()

    enhanced, _ = scan_page(photo, None)

    assert enhanced.shape[:2] == photo.shape[:2]


def test_binarize_produces_only_black_and_white():
    gradient = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (200, 1))
    assert set(np.unique(binarize(enhance(gradient)))).issubset({0, 255})


def test_score_prefers_a_bright_smooth_region_over_a_dark_one():
    """The scoring weights brightness, which is what stops a dark tray from
    outranking the page when both are plausible quadrilaterals."""
    gray = np.full((400, 400), 40, np.uint8)
    gray[20:180, 20:180] = 230   # bright candidate
    gray[220:380, 220:380] = 60  # dark candidate, same size
    mask = np.zeros_like(gray)
    area = gray.size

    bright = np.array([[20, 20], [180, 20], [180, 180], [20, 180]], np.float32)
    dark = np.array([[220, 220], [380, 220], [380, 380], [220, 380]], np.float32)

    assert _score_contour(bright, mask, gray, area) > _score_contour(dark, mask, gray, area)


def test_score_penalises_a_textured_region_against_a_smooth_one():
    """Texture penalty is why a patterned cloth loses to a plain page."""
    gray = np.full((400, 400), 40, np.uint8)
    gray[20:180, 20:180] = 200                      # smooth bright candidate
    gray[220:380, 220:380] = 200
    gray[220:380:6, 220:380] = 20                   # same brightness, but striped
    mask = np.zeros_like(gray)
    area = gray.size

    smooth = np.array([[20, 20], [180, 20], [180, 180], [20, 180]], np.float32)
    textured = np.array([[220, 220], [380, 220], [380, 380], [220, 380]], np.float32)

    assert _score_contour(smooth, mask, gray, area) > _score_contour(textured, mask, gray, area)


def test_find_document_contour_without_gray_still_returns_a_quad():
    """gray is optional; without it brightness/texture scoring is skipped."""
    mask = np.zeros((300, 300), np.uint8)
    cv2.rectangle(mask, (40, 40), (260, 260), 255, -1)

    assert find_document_contour(mask, 300 * 300, None) is not None
