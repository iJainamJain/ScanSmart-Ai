"""Histogram analysis for before/after enhancement comparison."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_histogram_comparison(before: np.ndarray, after: np.ndarray, output_path: str | Path) -> None:
    """Plot grayscale intensity histograms of before/after images side by side.

    A flat or narrow histogram indicates low contrast; a wider spread after
    enhancement is the visual evidence that CLAHE/contrast adjustment
    actually did something, not just a claim.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, image, title in zip(axes, (before, after), ("Before", "After")):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist)
        ax.set_title(title)
        ax.set_xlim([0, 256])
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)
