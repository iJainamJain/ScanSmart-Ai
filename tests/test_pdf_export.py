from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from pypdf import PdfReader

from src.pdf.export import (
    export_multi_page_pdf,
    export_searchable_multi_page_pdf,
    export_searchable_pdf,
    export_single_page_pdf,
)


def _page(tmp_path, name, shade=255, size=(600, 400)):
    path = tmp_path / name
    cv2.imwrite(str(path), np.full(size, shade, dtype=np.uint8))
    return path


def test_export_single_page_pdf_creates_a_valid_pdf_file(tmp_path):
    pdf_path = tmp_path / "scan.pdf"

    export_single_page_pdf(_page(tmp_path, "page.png"), pdf_path)

    assert pdf_path.read_bytes()[:4] == b"%PDF"
    assert len(PdfReader(str(pdf_path)).pages) == 1


def test_export_multi_page_pdf_writes_one_page_per_image(tmp_path):
    images = [_page(tmp_path, f"p{i}.png", shade=200 + i) for i in range(3)]
    pdf_path = tmp_path / "multi.pdf"

    count = export_multi_page_pdf(images, pdf_path)

    assert count == 3
    assert len(PdfReader(str(pdf_path)).pages) == 3


def test_multi_page_pdf_respects_the_given_page_order(tmp_path):
    """Ordering is just the caller's list order, which is what lets a UI
    offer reorder/delete without this module knowing about either."""
    a, b = _page(tmp_path, "a.png", 250), _page(tmp_path, "b.png", 100)

    forward = tmp_path / "fwd.pdf"
    reverse = tmp_path / "rev.pdf"
    export_multi_page_pdf([a, b], forward)
    export_multi_page_pdf([b, a], reverse)

    assert forward.read_bytes() != reverse.read_bytes()


def test_multi_page_pdf_rejects_an_empty_page_list(tmp_path):
    with pytest.raises(ValueError):
        export_multi_page_pdf([], tmp_path / "empty.pdf")


def test_single_page_export_fails_loudly_on_an_unreadable_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_single_page_pdf(tmp_path / "missing.png", tmp_path / "out.pdf")


def test_searchable_pdf_returns_false_when_tesseract_is_unavailable(tmp_path):
    """Tesseract is an optional external binary, so its absence must degrade
    to the image-only PDF rather than break the export."""
    page = _page(tmp_path, "page.png")
    out = tmp_path / "ocr.pdf"

    with patch("src.pdf.export.pytesseract.image_to_pdf_or_hocr", side_effect=OSError("not installed")):
        assert export_searchable_pdf(page, out) is False

    assert not out.exists(), "a failed OCR run must not leave a partial file"


def test_searchable_multi_page_returns_false_when_tesseract_is_unavailable(tmp_path):
    pages = [_page(tmp_path, f"p{i}.png") for i in range(2)]
    out = tmp_path / "ocr_multi.pdf"

    with patch("src.pdf.export.pytesseract.image_to_pdf_or_hocr", side_effect=OSError("not installed")):
        assert export_searchable_multi_page_pdf(pages, out) is False

    assert not out.exists()


def test_searchable_multi_page_merges_one_pdf_per_image(tmp_path):
    """Tesseract emits a standalone PDF per image, so they have to be merged."""
    pages = [_page(tmp_path, f"p{i}.png") for i in range(3)]
    out = tmp_path / "merged.pdf"

    single = tmp_path / "one.pdf"
    export_single_page_pdf(pages[0], single)
    fake_page_pdf = single.read_bytes()

    with patch("src.pdf.export.pytesseract.image_to_pdf_or_hocr", return_value=fake_page_pdf):
        assert export_searchable_multi_page_pdf(pages, out) is True

    assert len(PdfReader(str(out)).pages) == 3
