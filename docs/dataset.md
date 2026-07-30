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
cover that, the team is capturing its own set of ~20–30 phone photos:

- Documents (receipts, notes, printed pages, assignments, bills,
  certificates, book pages) photographed on varied surfaces/backgrounds.
- Deliberate variation: tilt/rotation, perspective angle, shadows/uneven
  lighting, different document sizes.
- Stored in `dataset/raw/`, named descriptively (e.g.
  `raw/receipt_shadow_01.jpg`, `raw/notes_tilted_03.jpg`) so the variation
  category is visible from the filename during evaluation.
- **License:** owned by the project team, used for educational purposes
  only within this course project.
- **Used for:** the core document-detection → perspective-transform →
  flattening pipeline stages, and as the primary demo set for lab
  evaluation.

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
