from pathlib import Path

import numpy as np
import pytest

from src.compression.compare import compare_compression, save_compression_report


@pytest.fixture
def sample_image():
    # A gradient, not flat noise - flat/random images defeat JPEG's DCT compression
    # and would make every format collapse to roughly the same tiny size.
    gradient = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (200, 1))
    return gradient


def test_compare_compression_writes_png_and_all_jpeg_qualities(tmp_path, sample_image):
    sizes = compare_compression(sample_image, tmp_path)

    assert (tmp_path / "compression_png.png").exists()
    assert "png" in sizes
    for quality in (95, 75, 50):
        assert (tmp_path / f"compression_jpeg_q{quality}.jpg").exists()
        assert f"jpeg_q{quality}" in sizes


def test_lower_jpeg_quality_produces_smaller_or_equal_file_size(tmp_path, sample_image):
    sizes = compare_compression(sample_image, tmp_path)

    assert sizes["jpeg_q50"] <= sizes["jpeg_q75"] <= sizes["jpeg_q95"]


def test_save_compression_report_writes_readable_summary(tmp_path):
    sizes = {"png": 10_000, "jpeg_q95": 4_000, "jpeg_q50": 1_000}
    report_path = tmp_path / "report.txt"

    save_compression_report(sizes, report_path)

    content = report_path.read_text()
    assert "png: 10,000 bytes" in content
    assert "jpeg_q50 is 10.00x smaller than PNG" in content
