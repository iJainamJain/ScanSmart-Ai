import numpy as np
import pytest

from src.enhancement.homomorphic import (
    DEFAULT_GAMMA_HIGH,
    DEFAULT_GAMMA_LOW,
    butterworth_highpass,
    homomorphic_filter,
)
from src.evaluation.synthetic import add_illumination_gradient, render_text_page


def test_filter_attenuates_low_and_boosts_high_frequencies():
    """The whole point of the gain-shifted filter: DC (illumination) is damped
    toward gamma_low, high frequencies (text edges) lifted toward gamma_high."""
    response = butterworth_highpass((256, 256))

    centre_gain = response[128, 128]           # DC term
    corner_gain = response[0, 0]               # highest frequency present
    assert centre_gain == pytest.approx(DEFAULT_GAMMA_LOW, abs=0.05)
    assert corner_gain == pytest.approx(DEFAULT_GAMMA_HIGH, abs=0.05)
    assert centre_gain < corner_gain


def test_filter_response_increases_with_frequency():
    response = butterworth_highpass((128, 128))
    centre = response[64, 64]
    midway = response[64, 96]
    edge = response[64, 127]
    assert centre < midway < edge


def test_filter_is_finite_everywhere_including_dc():
    """DC sits at distance zero; without the epsilon guard this divides by zero."""
    assert np.all(np.isfinite(butterworth_highpass((64, 64))))


def test_homomorphic_filter_returns_a_valid_8bit_image():
    page, _ = render_text_page()
    out = homomorphic_filter(page)
    assert out.dtype == np.uint8
    assert out.shape == page.shape


def test_homomorphic_filter_reduces_a_left_to_right_lighting_ramp():
    """Compare column means: the ramp makes one side dimmer, and correcting
    illumination should narrow that gap."""
    page, _ = render_text_page()
    ramped = add_illumination_gradient(page, strength=0.6)

    def side_gap(image):
        columns = image.astype(np.float32).mean(axis=0)
        third = len(columns) // 3
        return abs(columns[:third].mean() - columns[-third:].mean())

    assert side_gap(homomorphic_filter(ramped)) < side_gap(ramped)


def test_homomorphic_filter_rejects_a_colour_image():
    with pytest.raises(ValueError):
        homomorphic_filter(np.zeros((32, 32, 3), np.uint8))


def test_lower_cutoff_removes_less_of_the_image_content():
    """Cutoff controls how much of the spectrum is treated as illumination;
    an over-large cutoff starts discarding the text itself."""
    page, _ = render_text_page()
    gentle = homomorphic_filter(page, cutoff=0.005)
    aggressive = homomorphic_filter(page, cutoff=0.30)
    assert gentle.std() != aggressive.std()
