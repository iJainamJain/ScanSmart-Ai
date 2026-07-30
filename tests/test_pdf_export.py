from pathlib import Path

import cv2
import numpy as np

from src.pdf.export import export_single_page_pdf


def test_export_single_page_pdf_creates_a_valid_pdf_file(tmp_path):
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.full((600, 400), 255, dtype=np.uint8))

    pdf_path = tmp_path / "scan.pdf"
    export_single_page_pdf(image_path, pdf_path)

    assert pdf_path.exists()
    assert pdf_path.read_bytes()[:4] == b"%PDF"
