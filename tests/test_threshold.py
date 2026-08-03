import pytest
import numpy as np

from src.segmentation.threshold import (
    adaptive_threshold,
    clean_mask,
    global_threshold,
    otsu_binarize,
    sauvola_threshold,
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


def test_sauvola_threshold_reads_a_stroke_under_a_lighting_gradient():
    gray = np.full((100, 100), 220, dtype=np.uint8)
    gray[:, 50:] = 140
    gray[40:60, 20:30] = 60
    gray[40:60, 70:80] = 60

    result = sauvola_threshold(gray)

    assert result[50, 25] == 0
    assert result[50, 75] == 0
    assert result[10, 10] == 255, "flat paper region must stay background"


def test_sauvola_resists_noise_that_defeats_a_plain_mean_threshold():
    """The whole point of weighting by local std: in a near-flat, mildly noisy
    region, a mean-only threshold (adaptive_threshold) can misread noise as
    ink, while Sauvola's std term keeps the threshold close to the mean there
    and stays quiet."""
    rng = np.random.default_rng(0)
    gray = np.full((120, 120), 200, dtype=np.uint8)
    noise = rng.integers(-8, 9, gray.shape)
    gray = np.clip(gray.astype(int) + noise, 0, 255).astype(np.uint8)

    mean_based = adaptive_threshold(gray, block_size=25, c=2)  # small C: easy to misfire
    sauvola = sauvola_threshold(gray, window_size=25, k=0.2)

    mean_ink_fraction = (mean_based == 0).mean()
    sauvola_ink_fraction = (sauvola == 0).mean()
    assert sauvola_ink_fraction < mean_ink_fraction, (
        "sauvola should call far less of this flat noisy region 'ink' than a "
        "low-margin mean threshold does"
    )
    assert sauvola_ink_fraction < 0.05


def test_sauvola_threshold_rejects_a_colour_image():
    with pytest.raises(ValueError):
        sauvola_threshold(np.zeros((40, 40, 3), np.uint8))


def test_adaptive_threshold_default_block_size_stays_at_the_validated_value():
    """Regression guard, not a behavioural test. block_size=91 was set after
    measuring fragmentation across the full 283-photo dataset (214 improved,
    15 regressed - visually confirmed as not real regressions - 54 tied
    against the previous default of 25). If this drifts back down without a
    similar full-batch check, real handwriting will fragment into
    disconnected dashes again - see the adaptive_threshold docstring."""
    import inspect

    default = inspect.signature(adaptive_threshold).parameters["block_size"].default
    assert default == 91


def test_sauvola_threshold_accepts_an_even_window_size():
    gray = np.full((50, 50), 200, dtype=np.uint8)
    gray[20:30, 20:30] = 40
    # Must not raise, and must still behave like the equivalent odd window.
    even = sauvola_threshold(gray, window_size=24)
    odd = sauvola_threshold(gray, window_size=25)
    assert even.shape == gray.shape
    assert np.array_equal(even, odd)
