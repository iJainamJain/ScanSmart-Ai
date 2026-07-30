# Handoff / Next Steps

Written for whichever AI session or teammate picks this up next. Read this
before touching the boundary-detection code — it captures hard-won lessons
from this session, not just a task list.

## Where things stand

- Repo: https://github.com/iJainamJain/ScanSmart-Ai
- Team: Jainam Jain (23108B0084), Dhanush Chowke (23108B0080), Vivek Jaiswal (23108B0082)
- Task 1 (proposal + GitHub link + dataset links) is submitted:
  `docs/SmartScan_AI_Project_Proposal.pdf`.
- The pipeline implements roadmap Phases 1–6 (see `docs/proposal.md`),
  fully working end-to-end, verified against all 31 real self-captured
  photos in `dataset/raw/` with zero crashes.
- 26 automated tests, all passing: `py -3.12 -m pytest -q` (or
  `.venv\Scripts\python.exe -m pytest -q`).

## Pipeline stages

Run via `py -3.12 main.py <image> [--corners "x,y x,y x,y x,y"]`. Each
stage below is saved to `outputs/<image-name>/`:

```
01_resized -> 02_grayscale -> 03_denoised -> 04_edges (Canny, demo only,
NOT used for detection) -> 05_paper_mask (Otsu) -> 06_cleaned_mask
(morph close+open) -> 07_document_boundary (preview overlay) ->
08_flattened (perspective transform) -> 09_contrast_adjusted ->
10_sharpened -> 11_enhanced (CLAHE) -> 12_histogram_comparison.png ->
13_global_threshold / 14_otsu_threshold / 15_adaptive_threshold
(comparison) -> 16_final_bw (adaptive + morph open/close — this is the
TRUE final output, not 11_enhanced) -> 17_compression/ (PNG vs JPEG
q95/75/50 + report.txt) -> 18_scan.pdf
```

`--corners` bypasses detection entirely with manually-specified points —
useful for debugging or when detection picks the wrong region.

## Dataset

- `dataset/raw/`: 31 self-captured photos (`jainam_doc_02`..`32`,
  committed to the repo — `jainam_doc_01` was a WhatsApp screenshot that
  got removed). Target is **300 total** (100 each from Jainam, Dhanush,
  Vivek) — Dhanush's and Vivek's aren't in yet as of this writing.
  Naming convention for new arrivals: `raw/<contributor>_<type>_<variation>_<nn>.jpg`
  (see `docs/dataset.md`).
- `dataset/external/`: two Kaggle datasets downloaded and extracted
  locally, **gitignored, not committed** (too large — 1.65GB + 8.7MB):
  - `scanned_images_ocr_vlm/` — 3,492 images, MIT license
  - `noisy_rotated_scanned_documents/` — 600 images, rotation-labeled
  - Re-download commands are in `README.md` if these go missing locally.

## Immediate next task: boundary-detection precision (not started)

Boundary detection is verified at **26/31 correct** (right region picked),
but two distinct problems remain, both in `src/detection/contours.py`'s
`find_document_contour`:

1. **Wrong-region failures (5/31: `jainam_doc_06, 08, 12, 18, 28`)** — a
   checkered cloth background in many photos out-brightens the actual
   page under Otsu thresholding, so `segment_paper` picks the cloth/tray
   instead. **`jainam_doc_08` is a genuinely unfixable case** (the page
   itself is cropped out of the photo's frame by the camera) — don't
   spend time on it, it needs a reshoot, not a code fix.

2. **Loose crops on "correct" detections** — discovered by binarizing the
   output (`16_final_bw.png`) and finding several detections counted as
   "correct" (`jainam_doc_05, 09, 13, 20, 24`, likely more) actually
   include a real strip of background inside the crop — invisible in the
   color `07_document_boundary.png` overlay, glaring once binarized.

### Planned approach

`find_document_contour` currently takes the **largest** 4-point contour
from the cleaned Otsu mask — first passable candidate wins, not the best
one. Replace this with a **scoring** approach: for every candidate
4-point contour (not just the largest), compute a weighted score from:

- **Brightness** (mean intensity inside the contour) — paper is usually bright.
- **Local texture/variance** (crosshatch penalty) — a *hard AND-mask*
  version of this was already tried and **reverted** (see commit
  `3bc816a`) because it fixed 2/5 wrong-region failures but broke 5
  previously-correct detections — net regression, 23/31 vs the shipped
  26/31. Use texture as a **soft weighted penalty** this time, not a hard
  mask veto that can erase real page pixels.
- **Rectangularity** (aspect ratio plausibility).
- **Shape regularity** (deviation from a true rectangle).

Pick the highest-scoring candidate instead of largest-area-first.

### Process rules for whoever does this (learned the hard way this session)

- **Verify against the actual final output** (`outputs/<name>/16_final_bw.png`),
  not an intermediate preview. A red boundary line on a color photo reads
  as "close enough" even when it isn't; binarization exposes any included
  background instantly. This is exactly how the "loose crop" problem
  above was found — it was invisible until this check.
- **Full-batch verification, never a spot check.** A 5-image spot check
  made the texture-suppression experiment look like a clear win before
  the complete 31-image recount caught that it was a net regression.
  Always re-run and re-check **all 31** photos before claiming an
  improvement — not a sample of 5 or 8.
- **Don't overfit tuning to these same 31 eyeballed images.** Once
  Dhanush/Vivek's photos land, hold some out as an independent check.
- **Success bar:** must not regress any of the current 26/31
  known-correct detections; should net-fix at least 3 of the 5
  wrong-region failures; should visibly tighten the loose crops when
  checked against `16_final_bw.png`.

### How to test any detection change

```bash
rm -rf outputs/jainam_doc_*
for f in dataset/raw/jainam_doc_*.jpg; do
  py -3.12 main.py "$f"
done
# Then visually inspect outputs/jainam_doc_NN/16_final_bw.png for ALL 31 — not a sample.
```

## After the detection-precision fix (no urgent plan needed yet)

- **Phase 7 (GUI / camera capture / preview+edit UI)** — deliberately
  deferred. Project principle: "do not prematurely build the GUI before
  the processing pipeline is reliable" — the detection fix above should
  land first. When it's time, a Streamlit or similar lightweight
  desktop/web UI is the recommended approach (see `docs/proposal.md`),
  **not a native mobile app** — mobile is explicitly out of scope for the
  graded semester deliverable (see the "Product vision" section in
  `docs/proposal.md`), even though the team's long-term product vision
  includes it.
- **Phase 8 (OCR, searchable PDF)** — optional/stretch, after Phase 7.
- Once Dhanush/Vivek's photos land (target: 300 total), merge into
  `dataset/raw/` with the naming convention above and re-run the full
  detection batch to get an updated accuracy number on the larger set.

## Git workflow notes

- **Always confirm with the user before `git push`** — established
  pattern this session, not enforced by tooling, just don't skip it.
- **Windows CRLF risk:** `.gitattributes` forces binary treatment for
  `*.pdf`/`*.png`/`*.jpg`/etc. Don't remove it — without it, git's
  line-ending conversion can silently corrupt committed binaries (this
  almost happened with the proposal PDF; caught via byte-count diff
  before committing).
- venv at `.venv/` (Python 3.12 via `py -3.12`), install via
  `pip install -r requirements.txt`.
