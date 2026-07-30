"""Synthetic document pages with known text, for validating OCR accuracy.

Capturing real photographs with reliable ground-truth text is slow, and this
project's own experience is that capture briefs come back not matching the
brief. Rendering the page instead gives exact ground truth for free, lets the
measurement harness be tested before anyone picks up a camera, and allows a
degradation to be applied in isolation - so an effect can be attributed to
one cause rather than guessed at.

The degradations mirror the conditions actually seen in dataset/raw: a
lighting gradient, a hard cast shadow, and the pre-printed ruling of a
notebook page.
"""

import cv2
import numpy as np

SAMPLE_LINES = [
    "The quick brown fox jumps over the lazy dog",
    "Digital Image Processing mini project",
    "Illumination reflectance model I equals R times L",
    "Adaptive thresholding handles uneven lighting",
    "Morphological closing estimates the background",
]


def render_text_page(
    lines: list[str] | None = None,
    width: int = 900,
    height: int = 1200,
    font_scale: float = 0.7,
    ink: int = 165,
    thickness: int = 2,
) -> tuple[np.ndarray, str]:
    """Render text onto a white page. Returns (image, ground_truth_text).

    These defaults were calibrated against the real dataset, not chosen by
    taste, and the ink level matters more than anything else here.

    Measured on real photos, handwriting sits at roughly 0.41-0.74 of the
    paper level (ink ~136 on paper ~189). An earlier version of this
    generator used ink=60 on paper 250 - a ratio of 0.24, far darker than any
    real page - and that single unrepresentative choice inverted the
    benchmark's conclusion: it reported that keeping the brightness/contrast
    lift after illumination flattening was best, when on real photos that
    combination erased most of the handwriting. At ink=165 (ratio 0.66) the
    benchmark agrees with the real data.

    Tesseract reads the *raw* rendered page at CER 0.004 whatever the size, so
    difficulty has to be set against the pipeline's own output: too bold and
    there is no headroom for a degradation to show any effect, too fine and
    binarisation erases the text entirely before the experiment begins.
    """
    lines = lines or SAMPLE_LINES
    page = np.full((height, width), 250, np.uint8)

    y = 110
    for line in lines:
        cv2.putText(
            page, line, (55, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, ink, thickness, cv2.LINE_AA
        )
        y += 62
    return page, "\n".join(lines)


def add_sensor_noise(page: np.ndarray, sigma: float = 9.0, seed: int = 0) -> np.ndarray:
    """Additive Gaussian noise, as from a phone sensor in low light."""
    rng = np.random.default_rng(seed)
    noisy = page.astype(np.float32) + rng.normal(0, sigma, page.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_ruled_lines(page: np.ndarray, spacing: int = 45, darkness: int = 205) -> np.ndarray:
    """Overlay faint horizontal ruling, like lined notebook paper.

    Drawn light on purpose: the ruling of real notebook paper is much fainter
    than ink, which is exactly why illumination flattening can remove it while
    leaving handwriting behind.
    """
    ruled = page.copy()
    for y in range(spacing, page.shape[0], spacing):
        cv2.line(ruled, (0, y), (page.shape[1], y), darkness, 1)
    return ruled


def add_illumination_gradient(page: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """Apply a smooth left-to-right falloff, as from an off-centre lamp."""
    width = page.shape[1]
    ramp = np.linspace(1.0, 1.0 - strength, width, dtype=np.float32)
    field = np.tile(ramp, (page.shape[0], 1))
    return np.clip(page.astype(np.float32) * field, 0, 255).astype(np.uint8)


def add_cast_shadow(page: np.ndarray, strength: float = 0.5, softness: int = 41) -> np.ndarray:
    """Darken a diagonal band, as from a hand or the phone itself."""
    height, width = page.shape[:2]
    mask = np.ones((height, width), np.float32)
    corners = np.array(
        [[0, int(height * 0.15)], [width, int(height * 0.45)],
         [width, int(height * 0.85)], [0, int(height * 0.55)]], np.int32
    )
    cv2.fillPoly(mask, [corners], 1.0 - strength)
    mask = cv2.GaussianBlur(mask, (softness | 1, softness | 1), 0)
    return np.clip(page.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def build_benchmark_page(
    ruled: bool = True,
    gradient: bool = True,
    shadow: bool = True,
    noise: bool = True,
) -> tuple[np.ndarray, str]:
    """A degraded page plus its ground-truth text, with each effect toggleable."""
    page, truth = render_text_page()
    if ruled:
        page = add_ruled_lines(page)
    if gradient:
        page = add_illumination_gradient(page)
    if shadow:
        page = add_cast_shadow(page)
    if noise:
        page = add_sensor_noise(page)
    return page, truth
