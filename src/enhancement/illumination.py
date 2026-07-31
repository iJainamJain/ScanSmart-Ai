"""Illumination flattening based on the illumination-reflectance model.

A photographed page is modelled as I = R * L: the observed image is the
reflectance R (the ink and paper - what we want) multiplied by an
illumination field L (lamp falloff, cast shadows, the darkening toward a
notebook's gutter). Recovering R = I / L removes the lighting while leaving
the content.

L is estimated with a grey-scale morphological *closing* rather than a blur.
Closing with a structuring element wider than any text stroke erases the dark
strokes and leaves the lit paper surface behind. Being non-linear, it holds a
hard shadow edge in place instead of smearing it across the boundary the way
a Gaussian would, so the shadow divides out cleanly rather than leaving a
halo.

Two measured properties worth knowing before changing anything here:

1. Denoising first is not optional. The division amplifies noise, which
   binarisation then removes together with the text. Without it, a noisy
   shadowed page got *worse* (CER 0.647 -> 0.812, characters read halved);
   with edge-preserving denoising it reaches 0.036.
2. Flattening is strongly *subtractive* at the binarisation stage - it removed
   60-75% of the ink pixels on a sample of the project's photos, and the
   survivors were an almost exact subset of the un-flattened output. Much of
   what goes is the notebook's pre-printed ruling (desirable), but faint
   pencil can go with it.

Because of (2), judge any change here with src/evaluation/ocr_accuracy.py
rather than by eye: a cleaner-looking page and a page with the faint writing
destroyed are visually similar and only ground-truth text separates them.
"""

import cv2
import numpy as np

DEFAULT_SE_FRACTION = 20
DEFAULT_MAX_GAIN = 4.0
DEFAULT_SMOOTHING = 31

BILATERAL_DIAMETER = 9
BILATERAL_SIGMA = 60


def estimate_illumination(
    gray: np.ndarray,
    se_fraction: int = DEFAULT_SE_FRACTION,
    smoothing: int = DEFAULT_SMOOTHING,
) -> np.ndarray:
    """Estimate the illumination field L of a grayscale page.

    The structuring element is sized relative to the image so it stays wider
    than the text regardless of resolution; anything narrower than it is
    treated as content and removed.
    """
    if gray.ndim != 2:
        raise ValueError("estimate_illumination expects a single-channel image")

    size = max(15, (gray.shape[1] // se_fraction) | 1)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, element)

    # A light blur removes blockiness left by the structuring element without
    # meaningfully softening the shadow edge the closing preserved.
    if smoothing > 1:
        closed = cv2.GaussianBlur(closed, (smoothing | 1, smoothing | 1), 0)
    return closed


def denoise_edge_preserving(gray: np.ndarray) -> np.ndarray:
    """Bilateral filter - smooths noise while keeping text edges sharp.

    This must run *before* the division, and the choice of filter is not
    incidental. The division amplifies whatever noise is present by up to
    `max_gain`, and binarisation then strips the amplified speckle along with
    the text. Measured on the synthetic benchmark, denoising first took a
    shadowed noisy page from CER 0.647 to 0.036, and a page with every
    degradation from 1.000 (nothing readable at all) to 0.045.

    Gaussian and median blurring were both measured as *worse* than no
    denoising on the clean-gradient case (0.121 and 0.089 against 0.049),
    because they soften the strokes the OCR depends on. Bilateral preserves
    edges while averaging within flat regions, which is the property needed.
    """
    return cv2.bilateralFilter(gray, BILATERAL_DIAMETER, BILATERAL_SIGMA, BILATERAL_SIGMA)


def flatten_illumination(
    gray: np.ndarray,
    max_gain: float = DEFAULT_MAX_GAIN,
    se_fraction: int = DEFAULT_SE_FRACTION,
    denoise: bool = True,
) -> np.ndarray:
    """Divide out the estimated illumination field, returning an evenly-lit page.

    `max_gain` caps the per-pixel amplification so that deep shadow, where the
    signal is weakest and the noise relatively strongest, is not blown up into
    speckle. `denoise` applies edge-preserving smoothing first; leaving it off
    is measurably harmful on noisy input (see denoise_edge_preserving).
    """
    source = denoise_edge_preserving(gray) if denoise else gray
    field = estimate_illumination(source, se_fraction).astype(np.float32) + 1.0
    gain = np.clip(255.0 / field, 0.0, max_gain)
    return np.clip(source.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def illumination_unevenness(gray: np.ndarray) -> float:
    """Spread of the illumination field over the paper, as a percentage of its mean.

    Useful for deciding whether flattening is worth applying at all. Note this
    must be measured on the ORIGINAL image: computing it on an already-flattened
    result is circular, since the division forces the field flat by construction.
    """
    field = estimate_illumination(gray).astype(np.float32)
    # >= rather than >, so a perfectly uniform field still selects pixels
    # instead of yielding an empty slice and a nan.
    values = field[field >= np.percentile(field, 40)]
    if values.size == 0:
        return 0.0
    return float(values.std() / max(values.mean(), 1e-6) * 100.0)
