"""PDF export for scanned documents: single-page, multi-page, and searchable (OCR)."""

import io
import os
import sys
from pathlib import Path

import cv2
import pytesseract
from pypdf import PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# Tesseract ships as a separate binary; pick it up from a common Windows
# install location when it isn't already on PATH.
if sys.platform == "win32":
    _WINDOWS_TESSERACT = r"D:\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_WINDOWS_TESSERACT):
        pytesseract.pytesseract.tesseract_cmd = _WINDOWS_TESSERACT

MARGIN = 1 * cm


def _draw_fitted(pdf: canvas.Canvas, image_path: str | Path) -> None:
    """Draw one image centred on the current A4 page, scaled to fit inside the margins."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    page_width, page_height = A4
    scale = min((page_width - 2 * MARGIN) / width, (page_height - 2 * MARGIN) / height)
    draw_width, draw_height = width * scale, height * scale
    pdf.drawImage(
        str(image_path),
        (page_width - draw_width) / 2,
        (page_height - draw_height) / 2,
        width=draw_width,
        height=draw_height,
    )


def export_single_page_pdf(image_path: str | Path, output_pdf_path: str | Path) -> None:
    """Create a one-page A4 PDF containing the scanned image, centred and fit to the page."""
    pdf = canvas.Canvas(str(output_pdf_path), pagesize=A4)
    _draw_fitted(pdf, image_path)
    pdf.save()


def export_multi_page_pdf(image_paths, output_pdf_path: str | Path) -> int:
    """Compile several scanned pages into one A4 PDF, in the order given.

    Page order is the caller's responsibility - reordering and deletion are
    just list operations on `image_paths`, so the UI can offer both without
    this module needing to know about either.
    """
    image_paths = list(image_paths)
    if not image_paths:
        raise ValueError("need at least one image to build a PDF")

    pdf = canvas.Canvas(str(output_pdf_path), pagesize=A4)
    for index, image_path in enumerate(image_paths):
        if index:
            pdf.showPage()
        _draw_fitted(pdf, image_path)
    pdf.save()
    return len(image_paths)


def export_searchable_pdf(image_path: str | Path, output_pdf_path: str | Path) -> bool:
    """Create a searchable PDF via Tesseract OCR; returns False if unavailable.

    Tesseract is an optional external dependency, so callers fall back to the
    image-only PDF rather than failing the export outright.
    """
    try:
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(str(image_path), extension="pdf")
    except Exception as error:  # pytesseract raises several unrelated types
        print(f"OCR unavailable ({error}); falling back to image-only PDF.", file=sys.stderr)
        return False

    Path(output_pdf_path).write_bytes(pdf_bytes)
    return True


def export_searchable_multi_page_pdf(image_paths, output_pdf_path: str | Path) -> bool:
    """Multi-page searchable PDF; returns False (writing nothing) if OCR is unavailable.

    Each page is OCR'd separately and the resulting one-page PDFs are merged,
    because Tesseract emits a standalone PDF per image.
    """
    image_paths = list(image_paths)
    if not image_paths:
        raise ValueError("need at least one image to build a PDF")

    writer = PdfWriter()
    try:
        for image_path in image_paths:
            page_pdf = pytesseract.image_to_pdf_or_hocr(str(image_path), extension="pdf")
            writer.append(io.BytesIO(page_pdf))
    except Exception as error:
        print(f"OCR unavailable ({error}); falling back to image-only PDF.", file=sys.stderr)
        return False

    with open(output_pdf_path, "wb") as handle:
        writer.write(handle)
    return True
