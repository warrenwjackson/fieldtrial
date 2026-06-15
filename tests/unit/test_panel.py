from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from fieldtrial.data.adapters import (
    CallablePanelAdapter,
    DataFramePanelAdapter,
    DuckDBTableAdapter,
    read_sql_query,
)
from fieldtrial.data.panel import GeoPanel
from fieldtrial.data.synthetic import (
    TreatmentInjection,
    generate_synthetic_panel,
    generate_synthetic_us_panel,
)
from fieldtrial.exceptions import ValidationError


def small_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geo_id": ["a", "a", "b", "b"],
            "date": ["2027-01-01", "2027-01-02", "2027-01-01", "2027-01-02"],
            "orders": [10, 12, 8, 9],
            "sessions": [100, 110, 90, 95],
        }
    )


def test_panel_validates_and_sorts():
    panel = GeoPanel.from_dataframe(small_panel())
    assert panel.markets == ["a", "b"]
    assert panel.validate().ok


def test_panel_detects_missing_cell():
    df = small_panel().iloc[:-1]
    panel = GeoPanel.from_dataframe(df, validate=False)
    result = panel.validate()
    assert not result.ok
    assert result.missing_cells == 1


def test_panel_rejects_missing_required_columns():
    with pytest.raises(ValueError):
        GeoPanel.from_dataframe(pd.DataFrame({"geo_id": ["a"]}))


def test_panel_parquet_roundtrip(tmp_path):
    panel = GeoPanel.from_dataframe(small_panel())
    path = panel.to_parquet(tmp_path / "panel.parquet")
    loaded = GeoPanel.from_parquet(path)
    assert loaded.df["orders"].sum() == 39


def test_panel_from_query_and_callable():
    df = small_panel()
    con = duckdb.connect(":memory:")
    con.register("panel", df)
    panel = GeoPanel.from_query(con, "select * from panel")
    assert panel.df.shape[0] == 4

    adapter = CallablePanelAdapter(lambda **kwargs: df)
    assert adapter.fetch().df.shape[0] == 4


def test_sql_query_adapter_does_not_retry_without_params():
    class ParamStyleMismatchConnection:
        def __init__(self) -> None:
            self.calls = []

        def execute(self, query, params=None):
            self.calls.append((query, params))
            if params is not None:
                raise RuntimeError("unsupported paramstyle")
            return pd.DataFrame(
                {
                    "geo_id": ["a", "b"],
                    "date": ["2027-01-01", "2027-01-01"],
                    "orders": [1, 99],
                }
            )

    connection = ParamStyleMismatchConnection()
    with pytest.raises(ValidationError):
        read_sql_query(
            connection,
            "select * from panel where geo_id = :geo",
            params={"geo": "a"},
        )

    assert connection.calls == [("select * from panel where geo_id = :geo", {"geo": "a"})]


def test_duckdb_table_adapter_quotes_table_and_column_identifiers():
    con = duckdb.connect(":memory:")
    con.register("source_panel", small_panel().rename(columns={"orders": "gross orders"}))
    con.execute('create table "panel table" as select * from source_panel')

    fetched = DuckDBTableAdapter(con, "panel table").fetch(columns=["gross orders"])

    assert fetched["gross orders"].sum() == 39


def test_panel_from_adapter_filters_bounded_metric_columns():
    panel = GeoPanel.from_adapter(
        DataFramePanelAdapter(small_panel()),
        geos=["a"],
        start="2027-01-02",
        end="2027-01-02",
        columns=["orders"],
    )

    assert panel.markets == ["a"]
    assert panel.metric_columns == ["orders"]
    assert panel.df["orders"].tolist() == [12]


def test_geopanel_aggregate_can_coarsen_by_frequency():
    rows = []
    for geo in ["a", "b"]:
        for index, dt in enumerate(pd.date_range("2027-01-01", periods=4, freq="D"), start=1):
            rows.append({"geo_id": geo, "date": dt, "orders": index})
    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)

    aggregated = panel.aggregate(["orders"], freq="2D")

    assert aggregated.columns.tolist() == ["geo_id", "date", "orders"]
    assert aggregated["date"].dt.strftime("%Y-%m-%d").unique().tolist() == [
        "2027-01-01",
        "2027-01-03",
    ]
    assert aggregated.loc[aggregated["geo_id"] == "a", "orders"].tolist() == [3, 7]


def test_default_frequency_accepts_complete_monthly_panel():
    dates = pd.date_range("2027-01-01", periods=3, freq="MS")
    df = pd.DataFrame(
        [{"geo_id": geo, "date": dt, "orders": 1} for geo in ["a", "b"] for dt in dates]
    )

    panel = GeoPanel.from_dataframe(df)
    diagnostics = panel.complete_grid_diagnostics()

    assert diagnostics.is_complete
    assert diagnostics.expected_rows == 6


def test_explicit_daily_frequency_reports_missing_days_for_weekly_panel():
    df = pd.DataFrame(
        [
            {"geo_id": geo, "date": dt, "orders": 1}
            for geo in ["a", "b"]
            for dt in pd.to_datetime(["2027-01-01", "2027-01-08"])
        ]
    )

    panel = GeoPanel.from_dataframe(df, validate=False, frequency="D")
    result = panel.validate()

    assert result.ok is False
    assert result.missing_cells == 12
    assert result.diagnostics is not None
    assert str(result.diagnostics.frequency) == "D"


def test_metric_columns_excludes_strings_and_explicit_metadata_only():
    df = generate_synthetic_us_panel(
        n_markets=4,
        start="2027-01-01",
        end="2027-01-07",
        seed=7,
    )
    metrics = GeoPanel.from_dataframe(df, require_complete_grid=False).metric_columns

    assert {"orders", "sessions", "revenue", "spend", "eligible_users"}.issubset(metrics)
    assert "population" in metrics
    assert "market_size" in metrics
    assert "region" not in metrics
    assert "state" not in metrics
    assert "latent_seasonality" not in metrics
    assert "market_effect" not in metrics
    assert "region_effect" not in metrics
    assert "common_shock" not in metrics
    assert "local_shock" not in metrics
    assert "treatment" not in metrics


def test_synthetic_generator_is_geography_neutral_by_default():
    df = generate_synthetic_panel(
        n_markets=4,
        start="2027-01-01",
        end="2027-01-07",
        seed=7,
        country="CA",
        grain="province",
        region_labels=["Atlantic", "Prairies"],
        geo_prefix="prov",
    )

    assert set(df["country"]) == {"CA"}
    assert set(df["geo_grain"]) == {"province"}
    assert set(df["region"]).issubset({"Atlantic", "Prairies"})
    assert set(df["geo_id"].str.extract(r"^([^_]+)_", expand=False)) == {"prov"}
    assert "state" not in df.columns


def test_synthetic_generator_is_deterministic():
    a = generate_synthetic_us_panel(n_markets=4, start="2027-01-01", end="2027-01-07", seed=7)
    b = generate_synthetic_us_panel(n_markets=4, start="2027-01-01", end="2027-01-07", seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_relative_count_lift_uses_balanced_rounding():
    geos = ["dma_001", "dma_002"]
    base = generate_synthetic_us_panel(
        n_markets=8,
        start="2027-01-01",
        end="2027-03-31",
        seed=11,
    )
    treated = generate_synthetic_us_panel(
        n_markets=8,
        start="2027-01-01",
        end="2027-03-31",
        seed=11,
        treatment=TreatmentInjection(
            geos=geos,
            start="2027-02-01",
            end="2027-02-28",
            lift=0.08,
            metric="orders",
        ),
    )
    mask = base["geo_id"].isin(geos) & base["date"].between("2027-02-01", "2027-02-28")

    base_total = int(base.loc[mask, "orders"].sum())
    treated_total = int(treated.loc[mask, "orders"].sum())

    assert treated_total == round(base_total * 1.08)


def test_synthetic_ratio_mode_injects_true_ratio_lift():
    geos = ["dma_001", "dma_002"]
    base = generate_synthetic_us_panel(
        n_markets=8,
        start="2027-01-01",
        end="2027-03-31",
        seed=21,
    )
    treated = generate_synthetic_us_panel(
        n_markets=8,
        start="2027-01-01",
        end="2027-03-31",
        seed=21,
        treatment=TreatmentInjection(
            geos=geos,
            start="2027-02-01",
            end="2027-02-28",
            lift=0.12,
            metric="conversion_rate",
            mode="ratio",
        ),
    )
    mask = base["geo_id"].isin(geos) & base["date"].between("2027-02-01", "2027-02-28")

    base_rate = base.loc[mask, "orders"].sum() / base.loc[mask, "sessions"].sum()
    treated_rate = treated.loc[mask, "orders"].sum() / treated.loc[mask, "sessions"].sum()

    assert treated.loc[mask, "sessions"].sum() == base.loc[mask, "sessions"].sum()
    tolerance = 0.5 / base.loc[mask, "orders"].sum()
    assert treated_rate / base_rate - 1 == pytest.approx(0.12, abs=tolerance)
