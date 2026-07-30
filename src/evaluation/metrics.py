"""Batch evaluation metrics for the scanning pipeline.

Produces the measurements the project proposal lists under "Evaluation
metrics": detection rate, processing time, output dimensions, and file size
before/after compression.

Note on what "detection success" means here: it records only that a
plausible quadrilateral was *found*, not that it was the correct one.
Whether the detected region is actually the page still has to be judged by
eye - every automatic proxy tried for that failed validation (see
docs/dataset.md). Treat this column as an upper bound.
"""

import csv
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from src.detection.contours import find_document_contour
from src.detection.edges import denoise, to_grayscale
from src.detection.refine import refine_quad
from src.enhancement.basic import enhance_document
from src.enhancement.contrast import adjust_brightness_contrast
from src.enhancement.sharpen import sharpen
from src.morphology.operations import closing, opening
from src.perspective.transform import four_point_transform
from src.preprocessing.loader import load_image, resize_image
from src.segmentation.threshold import adaptive_threshold, clean_mask, segment_paper


@dataclass
class ImageMetrics:
    image: str
    quad_found: bool
    quad_area_fraction: float
    seconds: float
    output_width: int
    output_height: int
    source_bytes: int
    png_bytes: int
    jpeg_q75_bytes: int


def evaluate_image(image_path: Path) -> ImageMetrics:
    """Run the pipeline on one image and measure it, without writing stage files."""
    started = time.perf_counter()
    original = load_image(image_path)
    resized, _ = resize_image(original)
    gray = to_grayscale(resized)
    blurred = denoise(gray)

    mask = clean_mask(segment_paper(blurred))
    image_area = resized.shape[0] * resized.shape[1]
    quad = find_document_contour(mask, image_area, gray)
    if quad is not None:
        quad = refine_quad(quad, gray)

    flattened = four_point_transform(resized, quad) if quad is not None else resized
    enhanced = enhance_document(sharpen(adjust_brightness_contrast(flattened, 10, 1.15)))
    final = closing(opening(adaptive_threshold(enhanced), kernel_size=3), kernel_size=3)
    elapsed = time.perf_counter() - started

    png_bytes = len(cv2.imencode(".png", enhanced)[1])
    jpeg_bytes = len(cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 75])[1])
    area_fraction = (
        float(cv2.contourArea(quad.astype(np.float32)) / image_area) if quad is not None else 0.0
    )

    return ImageMetrics(
        image=image_path.name,
        quad_found=quad is not None,
        quad_area_fraction=round(area_fraction, 4),
        seconds=round(elapsed, 3),
        output_width=final.shape[1],
        output_height=final.shape[0],
        source_bytes=image_path.stat().st_size,
        png_bytes=png_bytes,
        jpeg_q75_bytes=jpeg_bytes,
    )


def evaluate_dataset(image_paths, csv_path: Path) -> list[ImageMetrics]:
    """Evaluate every image and write a per-image CSV."""
    results = [evaluate_image(Path(p)) for p in image_paths]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(ImageMetrics)])
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))
    return results


def summarise(results: list[ImageMetrics]) -> str:
    """Human-readable summary of a batch, grouped by contributor prefix."""
    if not results:
        return "No images evaluated."

    lines = [f"Images evaluated: {len(results)}"]
    found = [r for r in results if r.quad_found]
    lines.append(f"Quadrilateral found: {len(found)}/{len(results)} ({len(found)/len(results):.0%})")
    lines.append("  (found != correct - correctness needs visual review, see module docstring)")

    times = [r.seconds for r in results]
    lines.append(f"Processing time: mean {np.mean(times):.2f}s, median {np.median(times):.2f}s, max {max(times):.2f}s")

    png = sum(r.png_bytes for r in results)
    jpeg = sum(r.jpeg_q75_bytes for r in results)
    source = sum(r.source_bytes for r in results)
    lines.append(f"Total source: {source/1e6:.1f}MB, enhanced as PNG: {png/1e6:.1f}MB, as JPEG q75: {jpeg/1e6:.1f}MB")
    if jpeg:
        lines.append(f"JPEG q75 is {png/jpeg:.2f}x smaller than PNG on the same output")

    lines.append("\nPer contributor:")
    for prefix in sorted({r.image.split("_")[0] for r in results}):
        subset = [r for r in results if r.image.startswith(prefix)]
        hits = sum(1 for r in subset if r.quad_found)
        mean_area = np.mean([r.quad_area_fraction for r in subset if r.quad_found] or [0])
        lines.append(
            f"  {prefix:<9} n={len(subset):<4} found={hits}/{len(subset)} "
            f"mean quad area={mean_area:.0%} of frame"
        )
    return "\n".join(lines)
