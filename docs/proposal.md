# Project Proposal — SmartScan AI: Intelligent Document Scanner

## Team

2-person Digital Image Processing mini project.

## Problem statement

Photos of physical documents taken with a phone camera are typically
tilted, perspective-distorted, unevenly lit, and cluttered with background.
SmartScan AI processes such a photo and produces a clean, flat,
scanner-quality digital document using classical image-processing
techniques, without relying on black-box ML models for the core pipeline.

## Objectives

- Automatically detect a document's boundary in a photo and correct its
  perspective.
- Enhance the flattened document (contrast, brightness, sharpness) so it is
  as readable as a real scan.
- Produce a clean, thresholded, scanner-like output ready for export.
- Demonstrate, in an inspectable and explainable way, the core DIP concepts
  taught this semester (see below).

## Pipeline

```
Input Image → Resizing → Preprocessing → Grayscale → Noise Reduction →
Edge Detection → Contour Detection → Document Boundary Detection →
Perspective Transform → Cropping/Flattening → Enhancement →
Thresholding/Binarization → Morphological Processing →
Final Scanner-Like Output → Export (Image/PDF)
```

## DIP concept mapping

| Concept area          | Techniques used |
|------------------------|------------------|
| Image enhancement      | Histogram analysis, histogram equalization, CLAHE, contrast/brightness adjustment, sharpening |
| Filtering               | Gaussian blur, median filtering, bilateral filtering |
| Segmentation            | Document/background separation, global & adaptive thresholding |
| Edge detection           | Canny edge detection |
| Morphological operations | Erosion, dilation, opening, closing |
| Geometric transforms      | Perspective transform, affine transforms, rotation |
| Image compression         | JPEG vs. PNG comparison, PDF size optimization |

## Development roadmap

| Phase | Focus |
|-------|-------|
| 1 | Repository setup, dataset setup, image loading, basic preprocessing |
| 2 | Grayscale, blur/filtering, Canny edges, contours, document boundary detection |
| 3 | Corner detection, point ordering, perspective transform, flattening |
| 4 | Histogram processing, contrast/CLAHE, sharpening, brightness correction |
| 5 | Global/adaptive/Otsu thresholding, erosion, dilation, opening, closing |
| 6 | JPEG/PNG comparison, compression, PDF generation, multi-page PDF |
| 7 | UI, image upload, camera capture, preview, processing controls, export |
| 8 (optional) | OCR, searchable PDF, document management, additional filters |

Phases 1–3 are implemented as of this MVP (see [README.md](../README.md)).

## Evaluation metrics

- Document detection success rate across the dataset
- Perspective correction quality (visual + geometric)
- Processing time per image
- Output image dimensions
- File size before/after compression
- OCR accuracy (once OCR is added)
- Visual before/after readability comparison

## Tech stack

Python, OpenCV, NumPy, Matplotlib, Pillow, scikit-image. Future: PyTesseract,
ReportLab/FPDF, a GUI framework (Streamlit/Tkinter/PySide).

## Dataset

See [dataset.md](dataset.md).
