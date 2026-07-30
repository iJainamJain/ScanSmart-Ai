"""Basic morphological operations for cleaning binarized document output."""

import cv2
import numpy as np


def _kernel(size: int) -> np.ndarray:
    return np.ones((size, size), np.uint8)


def erode(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Shrink white regions - removes small white speckle noise."""
    return cv2.erode(image, _kernel(kernel_size), iterations=iterations)


def dilate(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Grow white regions - thickens thin strokes, fills small black gaps."""
    return cv2.dilate(image, _kernel(kernel_size), iterations=iterations)


def opening(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Erode then dilate - removes small white speckle noise without shrinking real strokes."""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, _kernel(kernel_size))


def closing(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Dilate then erode - fills small black holes/gaps in strokes without growing them."""
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, _kernel(kernel_size))
