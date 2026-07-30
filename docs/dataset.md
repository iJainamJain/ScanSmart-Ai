# Dataset

The project uses two complementary sources: two large public Kaggle
datasets for the enhancement/segmentation/morphology/compression stages,
plus a self-captured dataset for the document-boundary-detection and
perspective-correction stage, which needs real photos of tilted documents
on cluttered backgrounds — a case the public sets below don't cover well.

## 1. Kaggle: Scanned Images Dataset for OCR and VLM finetuning

- **Link:** https://www.kaggle.com/datasets/suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning
- **Size:** 3,492 images, 1.82 GB
- **License:** MIT (freely reusable, including modification/redistribution)
- **Content:** 10 real-world document categories — advertisements, emails,
  forms, letters, memos, news, notes, reports, resumes, scientific papers.
  Flat scanned pages with varying quality, noise, and layout complexity.
- **Used for:** histogram/CLAHE enhancement, global/adaptive/Otsu
  thresholding, morphological operations, JPEG/PNG compression comparison.

## 2. Kaggle: Noisy and Rotated Scanned Documents

- **Link:** https://www.kaggle.com/datasets/sthabile/noisy-and-rotated-scanned-documents
- **Size:** 600 scanned images (500 with ground-truth rotation angle labels)
- **License:** Data files © Original Authors — attribution required, no
  redistribution of the raw files implied beyond course use.
- **Content:** Scanned pages captured at non-vertical angles with speckle
  noise.
- **Used for:** rotation/skew correction and denoising, with quantitative
  angle-accuracy evaluation against the provided ground-truth labels.

## 3. Self-captured dataset (own images)

Public datasets above are flat, pre-scanned pages — they don't have the
tilted-photo-on-a-cluttered-background conditions that document boundary
detection and perspective correction are actually built to handle. To
cover that, the team is capturing its own set:

- **Delivered: 283 images** — `jainam_doc_*` (31), `vivek_doc_*` (104),
  `dhanush_doc_*` (148). All stored at 1600px on the long side; the
  originals from Vivek and Dhanush were 8–12MP, downscaled on ingest
  because the pipeline caps at 1500px anyway and full-resolution copies
  would have added ~750MB to the repository for no functional gain.
- **License:** owned by the project team, used for educational purposes
  only within this course project.
- **Naming convention:** `raw/<contributor>_doc_<nn>.jpg`. The contributor
  prefix avoids collisions when merging three people's uploads.

### Framing problem — read before using this set for detection

A large portion of the delivered photos are **close-ups in which the page
fills the entire frame, with no visible background or page border**. Document
boundary detection and perspective correction cannot work on these: there is
no boundary in the pixels to find, so the detector returns a spurious
quadrilateral cutting across the page.

Assessed visually from contact sheets (see "how to review" below); these are
estimates, not exact counts — three attempts to classify this automatically
(texture contamination, crop area fraction, border-brightness ratio) were all
built and then **discarded after failing validation** against
visually-labelled images, so no reliable automatic measure is claimed:

| Set | Usable for boundary detection | Note |
|-----|-------------------------------|------|
| `jainam_doc_*` (31) | ~30 | Good framing; `jainam_doc_08` is cropped |
| `vivek_doc_*` (104) | ~65 (roughly `044`+) | First ~40 are close-ups |
| `dhanush_doc_*` (148) | few | Nearly all close-ups |

The close-up images are **still valid** for the enhancement, thresholding,
morphology, compression and OCR stages, which operate on an already-flat
page — a frame-filling page is effectively pre-flattened. They are kept in
the dataset for that reason rather than deleted.

**If more detection images are needed**, the reshoot brief is: place the
document on a contrasting surface, step back so all four corners and a
margin of background are inside the frame, and shoot at an angle.

**How to review framing:** build a contact sheet rather than opening images
one at a time — it makes reviewing the whole set a single glance, and is how
the above was assessed. See `docs/NEXT_STEPS.md`.

## Layout

```
dataset/
├── raw/         # Self-captured source images (see section 3 above)
├── external/    # Downloaded Kaggle datasets (gitignored, see below)
└── processed/   # Pipeline outputs kept for evaluation/comparison
```

The two Kaggle datasets are referenced by link and downloaded into
`dataset/external/` rather than committed to the repository, given their
size (1.82 GB + supporting files); download instructions are in the
[README](../README.md).
