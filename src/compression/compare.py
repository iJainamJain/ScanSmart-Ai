"""JPEG vs PNG compression comparison."""

from pathlib import Path

import cv2
import numpy as np

JPEG_QUALITIES = (95, 75, 50)


def compare_compression(image: np.ndarray, output_dir: Path) -> dict[str, int]:
    """Save `image` as PNG and JPEG at a few quality levels; return sizes in bytes.

    PNG is lossless, so it's the size baseline. Saving the same image as
    JPEG at several quality settings makes the quality/size tradeoff a
    measured fact rather than a claim.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}

    png_path = output_dir / "compression_png.png"
    cv2.imwrite(str(png_path), image)
    sizes["png"] = png_path.stat().st_size

    for quality in JPEG_QUALITIES:
        jpeg_path = output_dir / f"compression_jpeg_q{quality}.jpg"
        cv2.imwrite(str(jpeg_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        sizes[f"jpeg_q{quality}"] = jpeg_path.stat().st_size

    return sizes


def save_compression_report(sizes: dict[str, int], output_path: Path) -> None:
    """Write a human-readable size/ratio comparison from compare_compression's output."""
    lines = [f"{name}: {size:,} bytes" for name, size in sizes.items()]

    baseline = sizes.get("png")
    if baseline:
        lines.append("")
        for name, size in sizes.items():
            if name != "png" and size:
                lines.append(f"{name} is {baseline / size:.2f}x smaller than PNG")

    output_path.write_text("\n".join(lines))
