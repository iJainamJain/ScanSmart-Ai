"""Homomorphic filtering - illumination correction in the frequency domain.

An alternative to the morphological approach in illumination.py, kept
deliberately so the two can be compared rather than one asserted to be better.

Both start from the same model, I = R * L: the observed image is reflectance
(ink and paper) times illumination (lamp falloff, shadows). Taking logarithms
turns that product into a sum,

    log I = log R + log L

which matters because a Fourier transform is linear: once the two are added
rather than multiplied, they can be separated by frequency. Illumination
varies slowly across a page (low frequency); reflectance - the text - changes
abruptly at every stroke edge (high frequency). Attenuating the low
frequencies and boosting the high ones therefore suppresses the lighting while
keeping the content. Exponentiating returns to intensity.

The filter itself is a Butterworth high-pass shifted to sit between two gains:

    H(u,v) = (gamma_high - gamma_low) * [1 / (1 + (D0 / D(u,v))^(2n))] + gamma_low

so low frequencies are multiplied by roughly gamma_low (< 1, attenuating
illumination) and high frequencies by roughly gamma_high (> 1, boosting
detail), with a smooth transition of order n at cutoff D0. Butterworth rather
than an ideal cutoff because a sharp edge in the frequency domain produces
ringing in the spatial domain.

The known trade-off against morphological estimation is that this is a linear
low-pass model of the illumination, so a *hard* shadow edge - a step, not a
gradient - has high-frequency content that the filter cannot assign to
illumination, and it survives as a visible seam. Grey-scale closing, being
non-linear, holds that edge. The comparison arm in ocr_eval.py measures how
much that actually costs.
"""

import cv2
import numpy as np

DEFAULT_CUTOFF = 0.10   # as a fraction of the image diagonal
DEFAULT_ORDER = 2
DEFAULT_GAMMA_LOW = 0.4
DEFAULT_GAMMA_HIGH = 1.6


def butterworth_highpass(
    shape: tuple[int, int],
    cutoff: float = DEFAULT_CUTOFF,
    order: int = DEFAULT_ORDER,
    gamma_low: float = DEFAULT_GAMMA_LOW,
    gamma_high: float = DEFAULT_GAMMA_HIGH,
) -> np.ndarray:
    """Build the gain-shifted Butterworth high-pass filter, centred (DC in the middle)."""
    rows, cols = shape
    centre_row, centre_col = rows / 2.0, cols / 2.0

    row_distance = (np.arange(rows) - centre_row).reshape(-1, 1) ** 2
    col_distance = (np.arange(cols) - centre_col).reshape(1, -1) ** 2
    distance = np.sqrt(row_distance + col_distance)

    cutoff_px = max(cutoff * np.sqrt(rows**2 + cols**2), 1e-6)
    # +epsilon on distance keeps the DC term (distance 0) finite.
    response = 1.0 / (1.0 + (cutoff_px / (distance + 1e-6)) ** (2 * order))
    return (gamma_high - gamma_low) * response + gamma_low


def homomorphic_filter(
    gray: np.ndarray,
    cutoff: float = DEFAULT_CUTOFF,
    order: int = DEFAULT_ORDER,
    gamma_low: float = DEFAULT_GAMMA_LOW,
    gamma_high: float = DEFAULT_GAMMA_HIGH,
) -> np.ndarray:
    """Even out illumination via log -> FFT -> Butterworth high-pass -> inverse -> exp."""
    if gray.ndim != 2:
        raise ValueError("homomorphic_filter expects a single-channel image")

    # +1 keeps log finite at true black.
    log_image = np.log1p(gray.astype(np.float32))

    spectrum = np.fft.fftshift(np.fft.fft2(log_image))
    filtered = spectrum * butterworth_highpass(
        gray.shape, cutoff, order, gamma_low, gamma_high
    )
    spatial = np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))

    result = np.expm1(spatial)
    # The gains rescale the range arbitrarily, so normalise back to 8-bit.
    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
