from __future__ import annotations

from fieldtrial import GeoPanel
from fieldtrial.data.synthetic import SyntheticTreatment, generate_synthetic_us_panel
from fieldtrial.design.specs import CompletedExperimentSpec
from fieldtrial.estimators.ensemble import analyze_completed_experiment
from fieldtrial.reports.analysis import render_analysis_report


def test_analysis_workflow(tmp_path):
    spec = CompletedExperimentSpec.model_validate(
        {
            "experiment_id": "x",
            "start_date": "2027-05-01",
            "end_date": "2027-05-21",
            "pre_period_start": "2027-02-01",
            "pre_period_end": "2027-04-30",
            "treatment_geos": ["dma_001", "dma_002"],
            "control_geos": ["dma_010", "dma_011", "dma_012"],
            "primary_metrics": ["orders"],
            "metrics": {"orders": {"type": "count", "column": "orders"}},
        }
    )
    df = generate_synthetic_us_panel(
        n_markets=16,
        start="2027-01-01",
        end="2027-06-01",
        seed=12,
        treatment=SyntheticTreatment(spec.treatment_geos, "2027-05-01", "2027-05-21", lift=0.1),
    )
    results = analyze_completed_experiment(
        GeoPanel.from_dataframe(df, require_complete_grid=False),
        spec,
        estimators=["did", "synthetic_control"],
    )
    assert len(results) == 2
    assert render_analysis_report(results, tmp_path / "analysis.html").exists()
