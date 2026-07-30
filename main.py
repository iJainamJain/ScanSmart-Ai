"""SmartScan AI - MVP pipeline entry point.

Runs the document-scanning pipeline on a single input image and saves
every intermediate stage to outputs/<image-name>/ so each DIP technique
can be inspected individually during lab evaluation.

Usage:
    py -3.12 main.py dataset/raw/sample.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.detection.contours import find_document_contour
from src.detection.edges import denoise, detect_edges, to_grayscale
from src.enhancement.basic import enhance_document
from src.perspective.transform import four_point_transform
from src.preprocessing.loader import load_image, resize_image
from src.segmentation.threshold import clean_mask, segment_paper

OUTPUT_ROOT = Path("outputs")


def save_stage(output_dir: Path, stage_name: str, image) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{stage_name}.png"), image)


def parse_corners(text: str) -> np.ndarray:
    """Parse '--corners' CLI input like 'x1,y1 x2,y2 x3,y3 x4,y4' into a 4x2 array."""
    points = [tuple(map(float, pair.split(","))) for pair in text.split()]
    if len(points) != 4:
        raise ValueError("--corners needs exactly 4 'x,y' points")
    return np.array(points, dtype=np.float32)


def run_pipeline(image_path: str, manual_corners: np.ndarray | None = None) -> Path:
    image_path = Path(image_path)
    output_dir = OUTPUT_ROOT / image_path.stem

    original = load_image(image_path)
    resized, _scale = resize_image(original)
    save_stage(output_dir, "01_resized", resized)

    gray = to_grayscale(resized)
    save_stage(output_dir, "02_grayscale", gray)

    blurred = denoise(gray)
    save_stage(output_dir, "03_denoised", blurred)

    edges = detect_edges(blurred)
    save_stage(output_dir, "04_edges", edges)

    paper_mask = segment_paper(blurred)
    save_stage(output_dir, "05_paper_mask", paper_mask)

    cleaned_mask = clean_mask(paper_mask)
    save_stage(output_dir, "06_cleaned_mask", cleaned_mask)

    image_area = resized.shape[0] * resized.shape[1]
    document_corners = manual_corners if manual_corners is not None else find_document_contour(cleaned_mask, image_area)

    if document_corners is not None:
        boundary_preview = resized.copy()
        cv2.drawContours(boundary_preview, [document_corners.astype(int)], -1, (0, 0, 255), 3)
        save_stage(output_dir, "07_document_boundary", boundary_preview)

        flattened = four_point_transform(resized, document_corners)
    else:
        print(
            f"Warning: no document boundary found for {image_path.name}; "
            "falling back to the resized image.",
            file=sys.stderr,
        )
        flattened = resized

    save_stage(output_dir, "08_flattened", flattened)

    enhanced = enhance_document(flattened)
    save_stage(output_dir, "09_final", enhanced)

    print(f"Done. Stages saved to {output_dir}/")
    return output_dir / "09_final.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartScan AI document scanning pipeline")
    parser.add_argument("image", help="Path to an input document image")
    parser.add_argument(
        "--corners",
        help="Manually specify document corners, bypassing auto-detection: "
        "'x1,y1 x2,y2 x3,y3 x4,y4' in resized-image pixel coordinates "
        "(any point order - they get sorted internally). Use this when "
        "detection picks the wrong region.",
    )
    args = parser.parse_args()

    manual_corners = parse_corners(args.corners) if args.corners else None
    run_pipeline(args.image, manual_corners=manual_corners)


if __name__ == "__main__":
    main()
