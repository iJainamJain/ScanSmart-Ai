import cv2
import numpy as np

from src.detection.refine import MIN_AREA_RATIO, refine_quad
from src.perspective.transform import order_points


def _page_on_background(page_box=(120, 120, 380, 480)):
    """A bright smooth 'page' on a darker textured 'cloth' background."""
    gray = np.full((600, 500), 90, dtype=np.uint8)
    gray[::12, :] = 30  # crosshatch pattern makes the background high-variance
    gray[:, ::12] = 30
    x0, y0, x1, y1 = page_box
    gray[y0:y1, x0:x1] = 215
    return gray


def test_refine_pulls_an_oversized_quad_back_onto_the_page():
    gray = _page_on_background()
    # Overshoots the real page (x>=120, y>=120) on the top and left edges, the
    # way Otsu merging with a bright background actually fails in practice.
    oversized = np.array([[95, 95], [380, 95], [380, 480], [95, 480]], dtype=np.float32)

    refined = order_points(refine_quad(oversized, gray))

    assert refined[:, 0].min() > 110, "left edge should move in toward the page"
    assert refined[:, 1].min() > 110, "top edge should move down toward the page"
    assert cv2.contourArea(refined) < cv2.contourArea(order_points(oversized))


def test_refine_declines_to_correct_an_extreme_overshoot():
    """Documents a real limitation: the area guard that protects against
    cutting into content also means very large overshoots go uncorrected."""
    gray = _page_on_background()
    huge = np.array([[40, 30], [470, 30], [470, 560], [40, 560]], dtype=np.float32)

    refined = refine_quad(huge, gray)

    assert np.allclose(refined, huge), "guard should reject rather than half-fix"


def test_refine_leaves_an_already_tight_quad_essentially_alone():
    gray = _page_on_background()
    tight = np.array([[122, 122], [378, 122], [378, 478], [122, 478]], dtype=np.float32)

    refined = order_points(refine_quad(tight, gray))

    before = cv2.contourArea(order_points(tight))
    after = cv2.contourArea(refined)
    assert after / before > 0.9


def test_refine_rejects_a_result_that_would_cut_away_page_content():
    """The area guard must veto over-aggressive snaps rather than lose content."""
    gray = _page_on_background()
    quad = np.array([[122, 122], [378, 122], [378, 478], [122, 478]], dtype=np.float32)

    refined = refine_quad(quad, gray)

    ratio = cv2.contourArea(order_points(refined)) / cv2.contourArea(order_points(quad))
    assert ratio >= MIN_AREA_RATIO


def test_refine_returns_original_when_image_gives_no_usable_edges():
    flat = np.full((400, 400), 200, dtype=np.uint8)
    quad = np.array([[50, 50], [350, 50], [350, 350], [50, 350]], dtype=np.float32)

    refined = refine_quad(quad, flat)

    assert refined.shape == (4, 2)
    assert np.all(np.isfinite(refined))
