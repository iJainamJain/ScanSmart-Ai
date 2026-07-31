"""Measure OCR accuracy with and without illumination flattening.

Two modes:

  Synthetic (default) - renders pages with known text and applies controlled
  degradations. Needs no capture session, and isolates one effect at a time.

      py -3.12 ocr_eval.py

  Real - scores photographs against ground-truth text. Point it at a folder
  holding `<name>.jpg` beside `<name>.gt.txt`:

      py -3.12 ocr_eval.py --real dataset/ocr_pairs
"""

import argparse
import glob
from pathlib import Path

import cv2

from src.enhancement.homomorphic import homomorphic_filter
from src.enhancement.illumination import flatten_illumination, illumination_unevenness
from src.evaluation.ocr_accuracy import (
    compare_variants,
    summarise_variants,
    tesseract_available,
)
from src.evaluation.synthetic import build_benchmark_page
from src.pipeline import binarize, enhance

REPORT_DIR = Path("outputs/evaluation")

# Each case isolates one degradation so an effect can be attributed, not guessed.
SYNTHETIC_CASES = {
    "clean": dict(ruled=False, gradient=False, shadow=False, noise=False),
    "ruled only": dict(ruled=True, gradient=False, shadow=False, noise=False),
    "gradient only": dict(ruled=False, gradient=True, shadow=False, noise=False),
    "shadow only": dict(ruled=False, gradient=False, shadow=True, noise=False),
    "shadow + noise": dict(ruled=False, gradient=False, shadow=True, noise=True),
    "everything": dict(ruled=True, gradient=True, shadow=True, noise=True),
}


def variants_for(gray):
    """The two pipelines under comparison, both taken through to final B&W.

    Three arms: the plain pipeline, morphological illumination flattening
    (what the pipeline actually uses), and homomorphic filtering as a
    frequency-domain alternative, kept so the two illumination-correction
    methods can be compared rather than one asserted to be better.

    Both corrected arms skip the brightness/contrast lift, which normalising
    the illumination has already made redundant - leaving it in saturates the
    page and erases faint ink (see src.pipeline.enhance).
    """
    return {
        "baseline": binarize(enhance(gray)),
        "flattened": binarize(
            enhance(flatten_illumination(gray), illumination_normalized=True)
        ),
        "homomorphic": binarize(
            enhance(homomorphic_filter(gray), illumination_normalized=True)
        ),
    }


def run_synthetic() -> str:
    blocks = []
    for case, options in SYNTHETIC_CASES.items():
        page, truth = build_benchmark_page(**options)
        results = compare_variants(variants_for(page), truth)
        blocks.append(
            f"--- {case} (illumination unevenness {illumination_unevenness(page):.1f}%) ---\n"
            + summarise_variants(results)
        )
    return "\n\n".join(blocks)


def run_real(folder: str) -> str:
    blocks = []
    for image_path in sorted(glob.glob(str(Path(folder) / "*.jpg"))):
        truth_path = Path(image_path).with_suffix("").with_suffix(".gt.txt")
        if not truth_path.exists():
            truth_path = Path(image_path).with_suffix(".gt.txt")
        if not truth_path.exists():
            print(f"skipping {Path(image_path).name}: no .gt.txt alongside it")
            continue

        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        truth = truth_path.read_text(encoding="utf-8")
        results = compare_variants(variants_for(gray), truth)
        blocks.append(
            f"--- {Path(image_path).name} "
            f"(unevenness {illumination_unevenness(gray):.1f}%) ---\n"
            + summarise_variants(results)
        )
    return "\n\n".join(blocks) if blocks else "No image/ground-truth pairs found."


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR accuracy: baseline vs illumination-flattened")
    parser.add_argument("--real", metavar="FOLDER", help="score real photos with .gt.txt files")
    args = parser.parse_args()

    if not tesseract_available():
        raise SystemExit(
            "Tesseract is not available, so OCR accuracy cannot be measured.\n"
            "Install it and ensure it is on PATH (see src/pdf/export.py for the "
            "Windows location this project already checks)."
        )

    report = run_real(args.real) if args.real else run_synthetic()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "ocr_accuracy.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWritten to {REPORT_DIR / 'ocr_accuracy.txt'}")
    print(
        "\nRead a negative CER delta as 'flattening helped'. A large drop in "
        "characters_read alongside a worse CER means ink was destroyed, not cleaned."
    )


if __name__ == "__main__":
    main()
