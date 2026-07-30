# Dataset

## Target

~20–50 sample document images covering realistic variation:

- Clean, flat document images
- Tilted / rotated documents
- Perspective-distorted documents (angled camera shots)
- Documents with shadows or uneven lighting
- Documents photographed on different backgrounds/surfaces
- Document types: receipts, handwritten notes, printed pages, assignments,
  bills, certificates, book pages

## Layout

```
dataset/
├── raw/         # Original, unmodified source images
└── processed/   # Pipeline outputs kept for evaluation/comparison
```

Raw images should be named descriptively, e.g.
`raw/receipt_shadow_01.jpg`, `raw/notes_tilted_03.jpg`, so variation
categories are visible from the filename during evaluation.

## Sources

- Self-captured photos (phone camera, varied lighting/angles/backgrounds).
- Public, copyright-free/open document image datasets, e.g.:
  - [SmartDoc](https://www.icst.pku.edu.cn/cpdp/sdac/index.htm) — camera-captured document images with ground-truth corners.
  - [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) — distorted document images for dewarping research.
  - [MIDV-500](https://arxiv.org/abs/1807.05786) — identity document images/videos.

This file will be updated with the exact images/subsets actually used and
their license terms once the dataset is assembled.

## Licensing

Each external image or subset added to `dataset/raw/` must be logged here
with: source, license, and any attribution required. Self-captured images
are owned by the project team and used for educational purposes only.
