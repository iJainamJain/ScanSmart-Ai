# SmartScan AI — Intelligent Document Scanner

A Digital Image Processing (DIP) mini project that turns a photo of a physical
document (tilted, shadowed, off-center) into a clean, scanner-like digital
image using classical image-processing techniques — no black-box ML models
for the core pipeline.

Built incrementally over a semester to demonstrate: image enhancement,
filtering/noise removal, segmentation, thresholding, edge detection,
morphological operations, geometric transforms, and image compression.

## Current status: Phase 1–4

The pipeline currently:

1. Loads an input image
2. Resizes it (longer side capped at 1500px)
3. Converts to grayscale
4. Denoises with Gaussian blur
5. Detects edges (Canny) - saved for inspection; actual boundary detection
   uses Otsu segmentation instead (see below)
6. Segments the paper region via Otsu thresholding and morphological
   cleanup (close + open), then finds its 4-point boundary
7. Applies a perspective transform to flatten the document (or accepts
   manually-specified corners via `--corners`, see Usage)
8. Adjusts brightness/contrast, sharpens (unsharp masking), then applies
   CLAHE for adaptive contrast
9. Saves a before/after histogram comparison
10. Saves every intermediate stage for inspection

Boundary detection is verified at 26/31 correct on real self-captured
photos (see [docs/dataset.md](docs/dataset.md)); known failure mode and
planned fix are tracked as a backlog item.

Not yet implemented (planned for later phases — see [docs/proposal.md](docs/proposal.md)):
final-output binarization (adaptive/Otsu thresholding as a B&W scan mode),
morphological cleanup of that output, compression comparison, PDF export,
GUI, OCR.

## Project structure

```
dip proj/
├── app/                # Future UI / camera capture code (empty for now)
├── src/
│   ├── preprocessing/  # Image loading, resizing
│   ├── detection/      # Grayscale, blur, Canny edges, contours
│   ├── perspective/    # Corner ordering, four-point warp
│   ├── enhancement/    # CLAHE / contrast / brightness
│   ├── segmentation/   # Thresholding (future)
│   ├── morphology/     # Erosion/dilation/opening/closing (future)
│   ├── compression/    # JPEG/PNG/PDF size comparison (future)
│   ├── ocr/            # Tesseract OCR (future)
│   └── pdf/            # PDF export (future)
├── dataset/
│   ├── raw/            # Self-captured document photos (see docs/dataset.md)
│   ├── external/        # Downloaded Kaggle datasets (gitignored)
│   └── processed/      # Pipeline outputs used for evaluation
├── outputs/             # Per-run intermediate stage images (gitignored)
├── tests/               # Unit tests per module
├── docs/                 # Proposal, dataset notes, weekly progress
├── notebooks/            # Exploratory notebooks for trying out techniques
├── requirements.txt
└── main.py               # CLI entry point that runs the full pipeline
```

## Setup

Requires Python 3.10+.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
py -3.12 main.py dataset/raw/sample.jpg
```

Each pipeline stage is saved to `outputs/<image-name>/` (e.g.
`01_resized.png`, `02_grayscale.png`, ... `11_final.png`, plus a
`12_histogram_comparison.png`) so individual DIP techniques can be
demonstrated separately during lab evaluation.

If detection picks the wrong region, override it manually:

```bash
py -3.12 main.py dataset/raw/sample.jpg --corners "50,40 900,60 880,1200 40,1180"
```

## Dataset

See [docs/dataset.md](docs/dataset.md) for full sources and licensing.
Summary: two public Kaggle datasets (for enhancement/segmentation/
morphology/compression) plus a self-captured set in `dataset/raw/` (for
document detection and perspective correction).

```bash
# requires a Kaggle account + API token (~/.kaggle/kaggle.json)
kaggle datasets download -d suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning -p dataset/external/scanned_images_ocr_vlm --unzip
kaggle datasets download -d sthabile/noisy-and-rotated-scanned-documents -p dataset/external/noisy_rotated_scanned_documents --unzip
```

(Alternatively, download the zips manually from the Kaggle dataset pages and
extract into those same two subfolders.)

## Roadmap

See [docs/proposal.md](docs/proposal.md) for the full phase-by-phase plan
(document detection → perspective correction → enhancement → segmentation &
morphology → compression & export → application UI → OCR).
