"""Thresholding techniques: paper-region segmentation and final-output binarization."""

import cv2
import numpy as np


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    """Binarize using Otsu's automatic global threshold.

    Otsu picks a threshold from the image's bimodal histogram instead of a
    fixed guess; combined with THRESH_BINARY, pixels brighter than that
    threshold are always mapped to 255 regardless of how much of the frame
    they cover, so no polarity correction is needed here.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def global_threshold(gray: np.ndarray, thresh: int = 127) -> np.ndarray:
    """Binarize using a single fixed threshold value for the whole image."""
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary


def adaptive_threshold(gray: np.ndarray, block_size: int = 91, c: int = 15) -> np.ndarray:
    """Binarize using a locally-computed threshold per pixel neighborhood.

    A single global threshold (fixed or Otsu) breaks down when lighting
    varies across the page - part of the page ends up all-black or
    all-white. Adaptive thresholding computes each pixel's threshold from
    its own neighborhood, which handles shadows/uneven lighting far better
    and is what actually produces a clean "scanned B&W page" look.

    block_size=91 replaces an earlier default of 25, which was never
    validated against real full-resolution photos and fragmented ordinary
    handwriting into disconnected dashes - a small window sits close enough
    to a stroke's own edge that local contrast swings wildly pixel to pixel.
    A first attempt to fix this changed block_size AND c together and looked
    like a wash across a 41-photo batch (13 better/10 worse/18 tied); that
    conflated two variables. Isolating block_size alone, with c held at its
    original 15, gives 30 better/4 worse/7 tied on the same batch, and the
    losses are minor (verified visually, not just by the fragmentation
    metric a batch this size is scored with).
    """
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def sauvola_threshold(
    gray: np.ndarray, window_size: int = 25, k: float = 0.2, r: float = 128.0
) -> np.ndarray:
    """Binarize using Sauvola's method - local adaptive thresholding built for documents.

    Plain adaptive thresholding (see adaptive_threshold above) computes each
    pixel's threshold from the local MEAN only. That is a generic OpenCV
    primitive, not something designed for document binarization, and it has a
    specific failure mode on real handwriting: with a fixed window size, it
    fires on every faint local variation regardless of how much real contrast
    is present, fragmenting continuous pen strokes into disconnected dashes.
    Measured on a real photo, that fragmentation is directly visible and was
    confirmed across a 41-photo batch: a fragmentation-based comparison against
    a wider window was a wash (13 photos better, 10 worse, 18 tied) rather
    than a fix, because the "right" fixed window size differs per photo.

    Sauvola's threshold instead weights the local mean by local standard
    deviation:

        T(x, y) = mean(x, y) * (1 + k * (std(x, y) / R - 1))

    In a near-uniform region (blank paper, low std) the threshold stays close
    to the local mean, so noise doesn't get misread as ink. Where there is
    real local contrast (a stroke edge, high std) the threshold shifts to
    separate ink from paper properly. This makes it self-adjusting to stroke
    contrast rather than needing a hand-tuned window size per photo, which is
    the actual property that was missing from the mean-based approach.

    `r` is the dynamic range of the standard deviation and is conventionally
    128 for 8-bit grayscale images (Sauvola & Pietikainen, 2000); `k` in
    [0.2, 0.5] is standard, with smaller k preserving more faint detail.
    """
    if gray.ndim != 2:
        raise ValueError("sauvola_threshold expects a single-channel image")

    size = window_size | 1  # must be odd
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, ddepth=-1, ksize=(size, size))
    mean_sq = cv2.boxFilter(g * g, ddepth=-1, ksize=(size, size))
    std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))

    threshold = mean * (1.0 + k * (std / r - 1.0))
    return np.where(g > threshold, 255, 0).astype(np.uint8)


def segment_paper(gray: np.ndarray) -> np.ndarray:
    """Binarize a grayscale image so the (bright) paper is foreground (255).

    Thin wrapper around otsu_binarize - kept as a separate name because its
    purpose (finding the page's boundary in a raw photo) is distinct from
    using the same technique to binarize the final scanner output.
    """
    return otsu_binarize(gray)


def clean_mask(mask: np.ndarray, close_size: int = 5, open_size: int = 21) -> np.ndarray:
    """Morphologically clean a binary mask before contour extraction.

    Closing (small kernel) fills small holes inside the page silhouette.
    Opening (larger kernel) strips thin spurious protrusions - e.g. a stray
    bright reflection bridging into the background - that would otherwise
    drag a convex-hull boundary estimate way outside the actual page.
    """
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    return opened
