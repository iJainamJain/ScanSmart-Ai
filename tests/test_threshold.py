import numpy as np

from src.segmentation.threshold import clean_mask, segment_paper


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
