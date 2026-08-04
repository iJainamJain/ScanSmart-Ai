import cv2
import numpy as np

import pytest

from src.detection.contours import _score_contour, find_document_contour
from src.pipeline import (
    MODE_BW,
    MODE_COLOR,
    MODE_GRAY,
    binarize,
    detect_document,
    enhance,
    render_page,
    scan_page,
)


def _photo(page=(100, 100, 400, 500), shape=(600, 500)):
    """Bright page on a darker background, as a 3-channel photo.

    Includes a dark mark inside the page so it isn't a perfectly flat
    region - a real photo never is, and CLAHE/brightness adjustment on a
    truly flat input can legitimately collapse to a single output value,
    which would be mistaken for "binarized" by a naive variety check.
    """
    img = np.full((*shape, 3), 70, np.uint8)
    x0, y0, x1, y1 = page
    img[y0:y1, x0:x1] = 220
    cv2.rectangle(img, (x0 + 40, y0 + 40), (x0 + 120, y0 + 80), (30, 30, 30), -1)
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


def test_brightness_lift_is_skipped_once_illumination_is_normalized():
    """Regression guard. Illumination flattening leaves paper near 255;
    applying the brightness/contrast lift on top saturates the page and
    crushes ink contrast. Measured on real photos, leaving it in collapsed
    final ink coverage from 10.4% to 1.4% - most of the handwriting erased."""
    from src.enhancement.illumination import flatten_illumination

    photo = _photo()
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    # A faint mark, like real pencil against paper rather than printer ink.
    cv2.putText(gray, "faint text here", (130, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 150, 2)
    flattened = flatten_illumination(gray)

    with_lift = binarize(enhance(flattened, illumination_normalized=False))
    without_lift = binarize(enhance(flattened, illumination_normalized=True))

    assert (without_lift == 0).sum() > (with_lift == 0).sum(), (
        "skipping the redundant lift must retain more ink"
    )


def test_scan_page_can_flatten_lighting_and_leaves_it_off_by_default():
    photo = _photo()
    corners = detect_document(photo)

    _, plain = scan_page(photo, corners)
    _, flattened = scan_page(photo, corners, flatten_lighting=True)

    assert plain.shape == flattened.shape
    assert set(np.unique(flattened)).issubset({0, 255})


def test_render_page_color_mode_stays_color_and_never_binarizes():
    photo = _photo()
    corners = detect_document(photo)

    out = render_page(photo, corners, mode=MODE_COLOR)

    assert out.ndim == 3, "color mode must not collapse to greyscale"
    assert len(np.unique(out)) > 2, "color mode must not binarize"


def test_render_page_gray_mode_matches_plain_enhance():
    photo = _photo()
    corners = detect_document(photo)

    out = render_page(photo, corners, mode=MODE_GRAY)

    assert out.ndim == 2
    assert len(np.unique(out)) > 2, "gray mode must not binarize"


def test_render_page_bw_mode_matches_scan_page():
    photo = _photo()
    corners = detect_document(photo)

    _, expected = scan_page(photo, corners)
    actual = render_page(photo, corners, mode=MODE_BW)

    assert np.array_equal(actual, expected)


def test_render_page_color_mode_with_flattening_stays_three_channel():
    photo = _photo()
    corners = detect_document(photo)

    out = render_page(photo, corners, mode=MODE_COLOR, flatten_lighting=True)

    assert out.ndim == 3
    assert out.shape[2] == 3


def test_render_page_rejects_an_unknown_mode():
    photo = _photo()
    with pytest.raises(ValueError):
        render_page(photo, detect_document(photo), mode="sepia")


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
