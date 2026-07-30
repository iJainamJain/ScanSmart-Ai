import csv

import cv2
import numpy as np
import pytest

from src.evaluation.contact_sheet import build_contact_sheet
from src.evaluation.metrics import evaluate_dataset, evaluate_image, summarise


@pytest.fixture
def sample_photo(tmp_path):
    """A bright page on a dark background, written to disk as a real JPEG."""
    img = np.full((600, 500, 3), 60, np.uint8)
    img[100:500, 90:410] = 225
    path = tmp_path / "jainam_doc_01.jpg"
    cv2.imwrite(str(path), img)
    return path


def test_evaluate_image_reports_timing_size_and_geometry(sample_photo):
    result = evaluate_image(sample_photo)

    assert result.image == "jainam_doc_01.jpg"
    assert result.seconds > 0
    assert result.output_width > 0 and result.output_height > 0
    assert result.source_bytes > 0
    assert result.png_bytes > 0 and result.jpeg_q75_bytes > 0


def test_jpeg_beats_png_on_photographic_content(tmp_path):
    """On real photo-like detail JPEG wins by a wide margin (~3.8x on the
    actual dataset)."""
    rng = np.random.default_rng(0)
    noise = rng.integers(120, 210, (600, 500, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(noise, (5, 5), 0)
    path = tmp_path / "jainam_doc_02.jpg"
    cv2.imwrite(str(path), img)

    result = evaluate_image(path)

    assert result.jpeg_q75_bytes < result.png_bytes


def test_png_can_beat_jpeg_on_flat_synthetic_content(sample_photo):
    """The size ordering is content-dependent, not a law: on a small, mostly
    uniform image PNG's lossless run-length coding beats JPEG, whose fixed
    header/quantisation-table overhead dominates. Worth pinning so the
    compression comparison isn't reported as 'JPEG is always smaller'."""
    result = evaluate_image(sample_photo)

    assert result.png_bytes < result.jpeg_q75_bytes


def test_evaluate_dataset_writes_one_csv_row_per_image(tmp_path, sample_photo):
    csv_path = tmp_path / "out" / "metrics.csv"

    results = evaluate_dataset([sample_photo], csv_path)

    assert len(results) == 1
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["image"] == "jainam_doc_01.jpg"


def test_summary_groups_by_contributor_prefix(sample_photo):
    results = [evaluate_image(sample_photo)]
    text = summarise(results)

    assert "Images evaluated: 1" in text
    assert "jainam" in text
    # The caveat must survive: a found quad is not a correct quad.
    assert "found != correct" in text


def test_summary_handles_empty_input():
    assert "No images" in summarise([])


def test_contact_sheet_tiles_images_into_a_grid(tmp_path):
    images = [(f"{i:02d}", np.full((80, 60, 3), i * 20, np.uint8)) for i in range(5)]

    out = build_contact_sheet(images, tmp_path / "sheet.png", columns=3, cell=100)

    assert out.exists()
    sheet = cv2.imread(str(out))
    assert sheet.shape[1] == 300, "3 columns of 100px"
    assert sheet.shape[0] == 200, "5 images across 3 columns needs 2 rows"


def test_contact_sheet_accepts_grayscale_images(tmp_path):
    images = [("g", np.full((50, 50), 200, np.uint8))]
    out = build_contact_sheet(images, tmp_path / "g.png", columns=1, cell=80)
    assert out.exists()


def test_contact_sheet_rejects_an_empty_batch(tmp_path):
    with pytest.raises(ValueError):
        build_contact_sheet([], tmp_path / "none.png")
