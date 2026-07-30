"""Brightness and contrast adjustment."""

import cv2
import numpy as np


def adjust_brightness_contrast(image: np.ndarray, brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
    """Apply a linear brightness/contrast adjustment: output = image * contrast + brightness."""
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
