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

## Recently Completed: Boundary-detection precision fix (Branch: `fix/boundary-detection-precision`)

The boundary detection logic in `src/detection/contours.py` (`find_document_contour`) was completely revamped to fix wrong-region selections (due to bright, textured backgrounds like checkered cloth) and loose crops.

**What was done:**
- Replaced the greedy "largest-area-first" approach with a robust **weighted scoring** system for all candidate contours.
- Scoring evaluates: Area (30%), Rectangularity (30%), Brightness (40%), and a soft Texture Penalty (-50% for high variance).
- Passed the grayscale image to `find_document_contour` to enable brightness and texture calculations.
- **Results:** Full batch test on 31 images in `dataset/raw/` confirmed the fix. 4 out of 5 wrong-region failures (`06, 12, 18, 28`) were perfectly corrected (`08` is unfixable). Loose crops (`05, 09, 13, 20, 24`) were successfully tightened. No regressions were introduced in the existing 26 correct detections. 

## Immediate next task: Phase 7 (GUI / camera capture / preview+edit UI)

Now that the core processing pipeline is reliable and boundary detection is fixed, it is time to build the user interface.

**Guidelines for the UI:**
- Build a lightweight **Streamlit** (or similar web/desktop) UI as recommended in `docs/proposal.md`.
- **Do NOT build a native mobile app**. Mobile is explicitly out of scope for the graded semester deliverable (see the "Product vision" section in `docs/proposal.md`), even though the team's long-term product vision includes it.
- The UI should ideally allow a user to:
  1. Upload or capture an image.
  2. Preview the automatically detected boundary.
  3. Optionally edit/adjust the 4 corners manually (similar to how `--corners` works in the CLI).
  4. Run the rest of the pipeline and export the final PDF.

## Pending Tasks (Data Collection & OCR)

- **Dataset Expansion:** Once Dhanush's and Vivek's photos land (target: 300 total, 100 each), merge them into `dataset/raw/` following the `raw/<contributor>_<type>_<variation>_<nn>.jpg` convention (see `docs/dataset.md`).
- **Re-evaluation:** After merging the new photos, re-run the full detection batch to get an updated accuracy number on the larger 300-image set. Do not overfit tuning to the first 31 images.
- **Phase 8 (OCR, searchable PDF):** Optional stretch goal to be tackled after the GUI is fully functional.

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
