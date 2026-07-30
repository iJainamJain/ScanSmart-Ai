# SmartScan AI — Intelligent Document Scanner

A Digital Image Processing (DIP) mini project that turns a photo of a physical
document (tilted, shadowed, off-center) into a clean, scanner-like digital
image using classical image-processing techniques — no black-box ML models
for the core pipeline.

Built incrementally over a semester to demonstrate: image enhancement,
filtering/noise removal, segmentation, thresholding, edge detection,
morphological operations, geometric transforms, and image compression.

## Current status: Phase 1–3 MVP

The pipeline currently:

1. Loads an input image
2. Resizes it (longer side capped at 1500px)
3. Converts to grayscale
4. Denoises with Gaussian blur
5. Detects edges (Canny)
6. Finds contours and the largest 4-point document boundary
7. Applies a perspective transform to flatten the document
8. Applies basic contrast enhancement (CLAHE)
9. Saves every intermediate stage for inspection

Not yet implemented (planned for later phases — see [docs/proposal.md](docs/proposal.md)):
adaptive/Otsu thresholding, morphological cleanup, compression comparison,
PDF export, GUI, OCR.

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
│   ├── raw/            # Original sample document photos
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
`01_resized.png`, `02_grayscale.png`, ... `08_final.png`) so individual DIP
techniques can be demonstrated separately during lab evaluation.

## Dataset

See [docs/dataset.md](docs/dataset.md) for full sources and licensing.
Summary: two public Kaggle datasets (for enhancement/segmentation/
morphology/compression) plus a self-captured set in `dataset/raw/` (for
document detection and perspective correction).

```bash
# requires a Kaggle account + API token (~/.kaggle/kaggle.json)
kaggle datasets download -d suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning -p dataset/external --unzip
kaggle datasets download -d sthabile/noisy-and-rotated-scanned-documents -p dataset/external --unzip
```

## Roadmap

See [docs/proposal.md](docs/proposal.md) for the full phase-by-phase plan
(document detection → perspective correction → enhancement → segmentation &
morphology → compression & export → application UI → OCR).
