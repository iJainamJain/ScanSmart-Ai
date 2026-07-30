from pathlib import Path

import numpy as np
import pytest

from main import parse_corners, run_pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "sample_synthetic.jpg"


def test_pipeline_runs_end_to_end_and_produces_final_output():
    final_path = run_pipeline(str(FIXTURE))
    assert final_path.exists()


def test_pipeline_accepts_manual_corners_and_skips_detection():
    manual_corners = np.array([[0, 0], [400, 0], [400, 600], [0, 600]], dtype=np.float32)
    final_path = run_pipeline(str(FIXTURE), manual_corners=manual_corners)
    assert final_path.exists()


def test_parse_corners_reads_four_xy_pairs():
    result = parse_corners("0,0 400,0 400,600 0,600")
    assert np.array_equal(result, np.array([[0, 0], [400, 0], [400, 600], [0, 600]], dtype=np.float32))


def test_parse_corners_rejects_wrong_point_count():
    with pytest.raises(ValueError):
        parse_corners("0,0 400,0 400,600")
