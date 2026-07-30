"""Grayscale conversion, noise reduction, and edge detection."""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Reduce noise with a Gaussian blur while keeping document edges usable.

    Gaussian blur is preferred over median here since it's cheap and smooths
    sensor/lighting noise without eroding the straight edges Canny relies on.
    """
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def detect_edges(gray: np.ndarray, low_threshold: int = 75, high_threshold: int = 200) -> np.ndarray:
    """Run Canny edge detection on a (blurred) grayscale image."""
    return cv2.Canny(gray, low_threshold, high_threshold)
