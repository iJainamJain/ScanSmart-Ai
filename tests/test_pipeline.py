from pathlib import Path

from main import run_pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "sample_synthetic.jpg"


def test_pipeline_runs_end_to_end_and_produces_final_output():
    final_path = run_pipeline(str(FIXTURE))
    assert final_path.exists()
