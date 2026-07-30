# SmartScan AI — Intelligent Document Scanner

A Digital Image Processing (DIP) mini project that turns a photo of a physical
document (tilted, shadowed, off-center) into a clean, scanner-like digital
image using classical image-processing techniques — no black-box ML models
for the core pipeline.

Built incrementally over a semester to demonstrate: image enhancement,
filtering/noise removal, segmentation, thresholding, edge detection,
morphological operations, geometric transforms, and image compression.

## Current status: Phase 1–6

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
10. Binarizes the enhanced page with global, Otsu, and adaptive
    thresholding (all three saved for comparison); adaptive thresholding
    feeds the final B&W scanner-style output
11. Cleans that binarized output with morphological opening + closing
12. Saves a JPEG-vs-PNG compression comparison (multiple JPEG quality
    levels) with a size/ratio report
13. Exports the final scan as a single-page A4 PDF
14. Saves every intermediate stage for inspection

Boundary detection is verified at 26/31 correct (right region picked) on
real self-captured photos, though several of those 26 include a looser
crop margin than ideal (visible once binarized) - see
[docs/dataset.md](docs/dataset.md) and the tracked backlog item for the
planned precision fix.

Not yet implemented (planned for later phases — see [docs/proposal.md](docs/proposal.md)):
GUI, camera capture, multi-page documents, OCR.

## Project structure

```
dip proj/
├── app/                # Future UI / camera capture code (empty for now)
├── src/
│   ├── preprocessing/  # Image loading, resizing
│   ├── detection/      # Grayscale, blur, Canny edges, contours
│   ├── perspective/    # Corner ordering, four-point warp
│   ├── enhancement/    # CLAHE / contrast / brightness / sharpening / histograms
│   ├── segmentation/   # Global / Otsu / adaptive thresholding
│   ├── morphology/     # Erosion/dilation/opening/closing
│   ├── compression/    # JPEG/PNG size comparison
│   ├── ocr/            # Tesseract OCR (future)
│   └── pdf/            # Single-page PDF export
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
`01_resized.png`, `02_grayscale.png`, ... `16_final_bw.png`, a
`12_histogram_comparison.png`, a `17_compression/` folder with a size
report, and an `18_scan.pdf`) so individual DIP techniques can be
demonstrated separately during lab evaluation.

If detection picks the wrong region, override it manually:

```bash
py -3.12 main.py dataset/raw/sample.jpg --corners "50,40 900,60 880,1200 40,1180"
```

## Evaluation

```bash
py -3.12 evaluate.py --sheet
```

Measures the whole dataset — detection rate, per-image processing time,
output dimensions, and file size before/after compression — writing
`outputs/evaluation/metrics.csv`, a printed summary, and (with `--sheet`)
a contact sheet of every detected boundary.

Read the detection rate carefully: it counts quadrilaterals **found**, not
quadrilaterals that are **correct**. On a photo with no visible page border
the detector returns a frame-sized quad rather than admitting failure, so
the reported rate is an upper bound. Correctness still needs a look at the
contact sheet.

## OCR accuracy (ground-truthed measurement)

```bash
py -3.12 ocr_eval.py                        # synthetic pages, known text
py -3.12 ocr_eval.py --real dataset/pairs   # photos with <name>.gt.txt alongside
```

Scores character/word error rate for the pipeline with and without
illumination flattening. This exists because "the output looks cleaner" is not
evidence — several invented image-quality metrics were discarded here after
failing to agree with human judgement, so enhancement claims are measured
against ground-truth text instead.

Current measured result (synthetic benchmark, lower CER is better):

| case | baseline CER | flattened CER | |
|------|-------------|---------------|---|
| clean | 0.143 | **0.054** | helps |
| ruled lines only | 0.129 | **0.067** | helps |
| lighting gradient | 0.071 | 0.080 | ~unchanged |
| cast shadow | 0.094 | 0.094 | no change |
| shadow + sensor noise | 0.661 | **0.161** | rescued |
| all degradations | **1.000** (unreadable) | **0.804** | 0 → 50 chars read |

On real photos, final ink coverage goes 10.0% → 11.3%, i.e. flattening
*recovers* strokes the baseline missed rather than removing them.

Two non-obvious steps were needed to get there, both found by measurement:

1. **Denoise first, with an edge-preserving filter.** The division amplifies
   noise, which binarisation then strips along with the text. Gaussian and
   median blur were both measured as *worse* than no denoising at all, since
   they soften the strokes OCR depends on; bilateral preserves them.
2. **Skip the brightness/contrast lift afterwards.** Flattening already
   leaves paper near 255, so the lift saturates the page and crushes ink
   contrast. Leaving it in collapsed real-photo ink coverage from 10.4% to
   **1.4%** — most of the handwriting erased, while still looking plausible.

Both are pinned by regression tests.

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

For the current work-in-progress item and exactly what to do next, see
[docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).
