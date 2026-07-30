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

from src.detection.contours import find_contours, find_document_contour
from src.detection.edges import denoise, detect_edges, to_grayscale
from src.enhancement.basic import enhance_document
from src.perspective.transform import four_point_transform
from src.preprocessing.loader import load_image, resize_image

OUTPUT_ROOT = Path("outputs")


def save_stage(output_dir: Path, stage_name: str, image) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{stage_name}.png"), image)


def run_pipeline(image_path: str) -> Path:
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

    contours = find_contours(edges)
    image_area = resized.shape[0] * resized.shape[1]
    document_corners = find_document_contour(contours, image_area)

    contour_preview = resized.copy()
    cv2.drawContours(contour_preview, contours, -1, (0, 255, 0), 1)
    save_stage(output_dir, "05_all_contours", contour_preview)

    if document_corners is not None:
        boundary_preview = resized.copy()
        cv2.drawContours(boundary_preview, [document_corners.astype(int)], -1, (0, 0, 255), 3)
        save_stage(output_dir, "06_document_boundary", boundary_preview)

        flattened = four_point_transform(resized, document_corners)
    else:
        print(
            f"Warning: no document boundary found for {image_path.name}; "
            "falling back to the resized image.",
            file=sys.stderr,
        )
        flattened = resized

    save_stage(output_dir, "07_flattened", flattened)

    enhanced = enhance_document(flattened)
    save_stage(output_dir, "08_final", enhanced)

    print(f"Done. Stages saved to {output_dir}/")
    return output_dir / "08_final.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartScan AI document scanning pipeline")
    parser.add_argument("image", help="Path to an input document image")
    args = parser.parse_args()

    run_pipeline(args.image)


if __name__ == "__main__":
    main()
