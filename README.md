# SmartScan AI — Intelligent Document Scanner

A Digital Image Processing (DIP) mini project that turns a photo of a physical
document (tilted, shadowed, off-center) into a clean, scanner-like digital
image using classical image-processing techniques — no black-box ML models
for the core pipeline.

Built incrementally over a semester to demonstrate: image enhancement,
filtering/noise removal, segmentation, thresholding, edge detection,
morphological operations, geometric transforms, and image compression.

## Current status: Phase 1–8

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
8. Removes shadows and lighting gradients by estimating the illumination
   field (grey-scale morphological closing) and dividing it out — on by
   default, disable with `--no-flatten`. The field itself is saved as a
   heat map, so the extracted shadow can be shown on its own.
9. Sharpens (unsharp masking), then applies CLAHE for adaptive contrast.
   The brightness/contrast lift is applied only when flattening is off,
   since flattening already normalises the paper level
10. Saves a before/after histogram comparison
11. Binarizes the enhanced page with global, Otsu, and adaptive
    thresholding (all three saved for comparison); adaptive thresholding
    feeds the final B&W scanner-style output
12. Cleans that binarized output with morphological opening + closing
13. Saves a JPEG-vs-PNG compression comparison (multiple JPEG quality
    levels) with a size/ratio report
14. Exports the final scan as a single-page A4 PDF
15. Saves every intermediate stage for inspection

Boundary detection is verified at 26/31 correct (right region picked) on
the original self-captured sample, though several of those 26 include a
looser crop margin than ideal (visible once binarized) - see
[docs/dataset.md](docs/dataset.md) and the tracked backlog item for the
planned precision fix.

### Adaptive threshold window size (fixed after a real bug report)

A live run on a real photo came back completely unreadable - fragmented into
disconnected dashes, not the shadow/lighting problem it first looked like.
Root-caused to `adaptive_threshold`'s window (`block_size`): the old default
of 25px sits close enough to a single stroke's own width that local contrast
swings pixel to pixel, fragmenting continuous pen strokes. It had never been
validated against real full-resolution photos - only against smaller-effect
cases where the problem didn't show.

The fix (`block_size=91`, `c` unchanged at 15) was measured, not guessed.
Sauvola thresholding was tried first as a more principled local-adaptive
method, but at an equivalent window it showed the *same* fragmentation - the
window size was the actual variable, not the algorithm. A first sweep that
changed window size and `c` together looked like a wash (13 better/10
worse/18 tied over 41 photos); isolating window size alone, full dataset
(283 photos): **214 better, 15 worse, 54 tied**. The 15 "worse" cases were
checked visually, not just trusted from the metric - none are real
regressions; the fragmentation metric undercounts on a couple of them
because non-text scribble marks pull its median down.

Also implemented: a Streamlit GUI with camera capture, manual corner
override and page reorder/delete; multi-page and searchable (OCR) PDF export;
and a measurement layer (batch metrics, contact sheets, OCR error rate).

## Project structure

```
dip proj/
├── app/                # Streamlit GUI (upload/camera, corner override, pages)
├── src/
│   ├── preprocessing/  # Image loading, resizing
│   ├── detection/      # Grayscale, blur, Canny edges, contours
│   ├── perspective/    # Corner ordering, four-point warp
│   ├── enhancement/    # CLAHE, contrast, sharpening, histograms, illumination
│   ├── segmentation/   # Global / Otsu / adaptive / Sauvola thresholding
│   ├── morphology/     # Erosion/dilation/opening/closing
│   ├── compression/    # JPEG/PNG size comparison
│   ├── ocr/            # Tesseract binary discovery
│   ├── pdf/            # Single-page, multi-page and searchable PDF export
│   └── evaluation/     # Batch metrics, contact sheets, OCR accuracy
├── dataset/
│   ├── raw/            # Self-captured document photos (see docs/dataset.md)
│   ├── external/        # Downloaded Kaggle datasets (gitignored)
│   └── processed/      # Pipeline outputs used for evaluation
├── outputs/             # Per-run intermediate stage images (gitignored)
├── tests/               # Unit tests per module
├── docs/                 # Proposal, dataset notes, weekly progress
├── notebooks/            # Exploratory notebooks for trying out techniques
├── requirements.txt
├── main.py               # CLI entry point that runs the full pipeline
├── evaluate.py           # Batch metrics + contact sheets
└── ocr_eval.py           # OCR error-rate benchmark
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

## Running the app

```bash
py -3.12 -m streamlit run app/main.py
```

Upload a photo or use the camera, correct the detected corners if needed, add
pages, and export a PDF (searchable when Tesseract is available). It works in
a phone browser and installs to the home screen.

Deployment instructions — including the one-off Streamlit Cloud setup — are in
[docs/DEPLOY.md](docs/DEPLOY.md).

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

### Two methods compared: morphological vs homomorphic

Illumination correction is implemented twice, deliberately, so the methods can
be compared rather than one asserted to be better:

- **Morphological** (`src/enhancement/illumination.py`) — estimate the
  illumination field with a grey-scale closing and divide it out. Non-linear,
  so it holds a hard shadow edge in place.
- **Homomorphic** (`src/enhancement/homomorphic.py`) — `log → FFT →
  gain-shifted Butterworth high-pass → inverse FFT → exp`. Taking logarithms
  turns `I = R·L` into a sum, which a linear transform can then separate by
  frequency.

Mean CER across the six benchmark cases: baseline **0.350**, morphological
**0.210**, homomorphic **0.369**.

The interesting part is *where* they differ. Homomorphic is **better at the
narrow job it was designed for** — a pure lighting gradient (0.031 vs 0.080)
or a cast shadow (0.040 vs 0.094) — but it boosts high frequencies by
construction, and noise is high-frequency, so it collapses entirely on noisy
pages (CER 1.000 where morphological reaches 0.161). Giving it the same
bilateral denoising rescues the noisy cases (1.000 → 0.295) but breaks clean
ones (ruled 0.094 → 1.000), showing it is unusually sensitive to how its input
is conditioned.

Morphological is therefore what the pipeline uses; homomorphic stays as a
measured comparison arm in `ocr_eval.py`.

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
