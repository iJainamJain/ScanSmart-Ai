import numpy as np

from src.segmentation.threshold import (
    adaptive_threshold,
    clean_mask,
    global_threshold,
    otsu_binarize,
    segment_paper,
)


def test_segment_paper_marks_bright_region_as_foreground_when_majority():
    gray = np.full((100, 100), 30, dtype=np.uint8)
    gray[10:90, 10:90] = 220  # bright page covers most of the frame

    mask = segment_paper(gray)

    assert mask[50, 50] == 255
    assert mask[2, 2] == 0


def test_segment_paper_marks_bright_region_as_foreground_when_minority():
    gray = np.full((100, 100), 30, dtype=np.uint8)
    gray[40:60, 40:60] = 220  # bright page is a small region against a larger dark background

    mask = segment_paper(gray)

    assert mask[50, 50] == 255, "the bright page must stay foreground even when it's the minority area"
    assert mask[2, 2] == 0


def test_clean_mask_removes_thin_spurious_protrusion():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 255  # main blob
    mask[0:50, 98:102] = 255  # thin 4px-wide spike reaching up to the border

    cleaned = clean_mask(mask)

    assert cleaned[100, 100] == 255, "the main blob must survive cleaning"
    assert cleaned[10, 100] == 0, "the thin spike must be removed by the opening step"


def test_global_threshold_splits_on_the_fixed_value():
    gray = np.array([[50, 100, 150, 200]], dtype=np.uint8)

    result = global_threshold(gray, thresh=127)

    assert list(result[0]) == [0, 0, 255, 255]


def test_otsu_binarize_separates_two_flat_regions():
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[:, :50] = 40
    gray[:, 50:] = 210

    result = otsu_binarize(gray)

    assert result[50, 10] == 0
    assert result[50, 90] == 255


def test_adaptive_threshold_handles_a_lighting_gradient_global_otsu_would_miss():
    # A page where the left half is bright (well-lit) and the right half is
    # dimmer (shadow), but text (darker strokes) exists on both halves.
    gray = np.full((100, 100), 220, dtype=np.uint8)
    gray[:, 50:] = 140  # shadowed half, still meant to read as "page background"
    gray[40:60, 20:30] = 60  # a text stroke on the bright side
    gray[40:60, 70:80] = 60  # a text stroke on the dim side, same absolute darkness gap

    result = adaptive_threshold(gray, block_size=25, c=15)

    assert result[50, 25] == 0, "text stroke on the bright side should read as foreground (dark)"
    assert result[50, 75] == 0, "text stroke on the shadowed side should also read as foreground (dark)"
