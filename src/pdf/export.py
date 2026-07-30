"""Single-page PDF export for the final scanned document."""

from pathlib import Path

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


def export_single_page_pdf(image_path: str | Path, output_pdf_path: str | Path) -> None:
    """Create a one-page A4 PDF containing the scanned image, centered and fit to the page."""
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]

    page_width, page_height = A4
    margin = 1 * cm
    available_width = page_width - 2 * margin
    available_height = page_height - 2 * margin
    scale = min(available_width / width, available_height / height)
    draw_width, draw_height = width * scale, height * scale
    x = (page_width - draw_width) / 2
    y = (page_height - draw_height) / 2

    pdf = canvas.Canvas(str(output_pdf_path), pagesize=A4)
    pdf.drawImage(str(image_path), x, y, width=draw_width, height=draw_height)
    pdf.save()
