"""Unsharp-mask sharpening."""

import cv2
import numpy as np


def sharpen(image: np.ndarray, amount: float = 0.6, blur_kernel: int = 5) -> np.ndarray:
    """Sharpen via unsharp masking: emphasize (original - blurred) detail.

    Blurring first isolates low-frequency content; subtracting it back out
    (weighted by `amount`) boosts edges without amplifying uniform areas.
    """
    blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
