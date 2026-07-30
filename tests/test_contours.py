import cv2
import numpy as np

from src.detection.contours import find_contours, find_document_contour


def _rectangle_mask(shape, top_left, bottom_right):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    return mask


def test_find_contours_finds_one_contour_for_one_rectangle():
    mask = _rectangle_mask((200, 200), (20, 20), (180, 180))

    contours = find_contours(mask)

    assert len(contours) == 1


def test_find_document_contour_returns_four_corners_for_a_clean_rectangle():
    shape = (200, 200)
    mask = _rectangle_mask(shape, (20, 20), (180, 180))
    image_area = shape[0] * shape[1]

    corners = find_document_contour(mask, image_area)

    assert corners is not None
    assert corners.shape == (4, 2)
    xs, ys = corners[:, 0], corners[:, 1]
    assert xs.min() == 20 and xs.max() == 180
    assert ys.min() == 20 and ys.max() == 180


def test_find_document_contour_returns_none_when_blob_too_small():
    shape = (200, 200)
    mask = _rectangle_mask(shape, (90, 90), (110, 110))  # 10% of the frame
    image_area = shape[0] * shape[1]

    assert find_document_contour(mask, image_area) is None


def test_find_document_contour_returns_none_for_empty_mask():
    mask = np.zeros((200, 200), dtype=np.uint8)

    assert find_document_contour(mask, 200 * 200) is None


def test_find_document_contour_recovers_four_corners_despite_a_small_notch():
    shape = (200, 200)
    mask = _rectangle_mask(shape, (20, 20), (180, 180))
    # simulate a curled page corner: a small bite out of one corner
    cv2.rectangle(mask, (20, 20), (35, 35), 0, -1)
    image_area = shape[0] * shape[1]

    corners = find_document_contour(mask, image_area)

    assert corners is not None
    assert corners.shape == (4, 2)
