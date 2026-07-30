"""Batch-evaluate the pipeline over a set of images.

Writes a per-image CSV, a printed summary, and (optionally) a contact sheet
of detected boundaries for visual review.

Usage:
    py -3.12 evaluate.py
    py -3.12 evaluate.py --pattern "dataset/raw/vivek_*.jpg" --sheet
"""

import argparse
import glob
from pathlib import Path

import cv2

from src.detection.contours import find_document_contour
from src.detection.edges import denoise, to_grayscale
from src.detection.refine import refine_quad
from src.evaluation.contact_sheet import build_contact_sheet
from src.evaluation.metrics import evaluate_dataset, summarise
from src.perspective.transform import order_points
from src.preprocessing.loader import load_image, resize_image
from src.segmentation.threshold import clean_mask, segment_paper

REPORT_DIR = Path("outputs/evaluation")


def boundary_preview(image_path: Path):
    """Resized image with the detected boundary drawn on it, for the contact sheet."""
    resized, _ = resize_image(load_image(image_path))
    gray = to_grayscale(resized)
    mask = clean_mask(segment_paper(denoise(gray)))
    quad = find_document_contour(mask, resized.shape[0] * resized.shape[1], gray)
    if quad is not None:
        quad = refine_quad(quad, gray)
        cv2.drawContours(resized, [order_points(quad).astype(int)], -1, (0, 0, 255), 4)
    return resized


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch pipeline evaluation")
    parser.add_argument("--pattern", default="dataset/raw/*.jpg", help="glob of images to evaluate")
    parser.add_argument("--sheet", action="store_true", help="also write a boundary contact sheet")
    parser.add_argument("--limit", type=int, default=0, help="evaluate at most N images (0 = all)")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No images matched {args.pattern}")

    print(f"Evaluating {len(paths)} images...")
    results = evaluate_dataset(paths, REPORT_DIR / "metrics.csv")

    summary = summarise(results)
    (REPORT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nCSV: {REPORT_DIR / 'metrics.csv'}")

    if args.sheet:
        previews = [(Path(p).stem.replace("_doc", ""), boundary_preview(Path(p))) for p in paths]
        sheet = build_contact_sheet(previews, REPORT_DIR / "boundaries.png")
        print(f"Contact sheet: {sheet}")


if __name__ == "__main__":
    main()
