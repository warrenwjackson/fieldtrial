"""Long-format geo panel abstraction."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fieldtrial.data.adapters import (
    CallablePanelAdapter,
    PanelAdapter,
    SQLQueryPanelAdapter,
    quote_sql_qualified_name,
    validate_sql_filter_clause,
)
from fieldtrial.data.validation import (
    CompleteGridDiagnostics,
    PanelValidationResult,
    complete_grid_diagnostics,
    infer_panel_frequency,
    require_columns,
    validate_long_panel,
)
from fieldtrial.exceptions import ValidationError


@dataclass
class GeoPanel:
    """Typed facade over a long-format geo-time metric panel."""

    df: pd.DataFrame
    geo_col: str = "geo_id"
    time_col: str = "date"
    frequency: str | pd.Timedelta | None = None
    market_metadata: pd.DataFrame | None = None
    _validation: PanelValidationResult | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        frame = self._normalize_frame(self.df, self.geo_col, self.time_col)
        self.df = frame
        if self.frequency is None:
            self.frequency = infer_panel_frequency(self.df[self.time_col])

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        *,
        geo_col: str = "geo_id",
        time_col: str = "date",
        frequency: str | pd.Timedelta | None = None,
        market_metadata: pd.DataFrame | None = None,
        validate: bool = True,
        require_complete_grid: bool = True,
        require_complete: bool | None = None,
    ) -> GeoPanel:
        if require_complete is not None:
            require_complete_grid = require_complete
        panel = cls(
            pd.DataFrame(df),
            geo_col=geo_col,
            time_col=time_col,
            frequency=frequency,
            market_metadata=market_metadata,
        )
        if validate:
            panel.validate(require_complete_grid=require_complete_grid).raise_if_invalid()
        return panel

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        geo_col: str = "geo_id",
        time_col: str = "date",
        frequency: str | pd.Timedelta | None = None,
        columns: Iterable[str] | None = None,
        validate: bool = True,
        require_complete_grid: bool = True,
        require_complete: bool | None = None,
        **read_kwargs: Any,
    ) -> GeoPanel:
        df = pd.read_parquet(
            path,
            columns=list(columns) if columns is not None else None,
            **read_kwargs,
        )
        return cls.from_dataframe(
            df,
            geo_col=geo_col,
            time_col=time_col,
            frequency=frequency,
            validate=validate,
            require_complete_grid=require_complete_grid,
            require_complete=require_complete,
        )

    @classmethod
    def from_duckdb(
        cls,
        database: str | Path,
        table: str,
        *,
        geo_col: str = "geo_id",
        time_col: str = "date",
        where: str | None = None,
        frequency: str | pd.Timedelta | None = None,
        validate: bool = True,
        require_complete_grid: bool = True,
    ) -> GeoPanel:
        import duckdb

        with duckdb.connect(str(database), read_only=True) as con:
            query = f"select * from {quote_sql_qualified_name(table)}"
            if where:
                query += f" where {validate_sql_filter_clause(where)}"
            return cls.from_query(
                con,
                query,
                geo_col=geo_col,
                time_col=time_col,
                frequency=frequency,
                validate=validate,
                require_complete_grid=require_complete_grid,
            )

    @classmethod
    def from_query(
        cls,
        connection: Any,
        query: str,
        *,
        params: Mapping[str, Any] | list[Any] | tuple[Any, ...] | None = None,
        geo_col: str = "geo_id",
        time_col: str = "date",
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        geos: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
        metrics: Iterable[str] | None = None,
        frequency: str | pd.Timedelta | None = None,
        validate: bool = True,
        require_complete_grid: bool = True,
        require_complete: bool | None = None,
    ) -> GeoPanel:
        adapter = SQLQueryPanelAdapter(
            connection=connection,
            query=query,
            geo_col=geo_col,
            time_col=time_col,
            params=params,
        )
        return cls.from_adapter(
            adapter,
            geo_col=geo_col,
            time_col=time_col,
            start=start,
            end=end,
            geos=geos,
            columns=columns,
            metrics=metrics,
            frequency=frequency,
            validate=validate,
            require_complete_grid=require_complete_grid,
            require_complete=require_complete,
        )

    @classmethod
    def from_callable(
        cls,
        fetcher: Callable[..., Any],
        *,
        geos: Iterable[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        metrics: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
        geo_col: str = "geo_id",
        time_col: str = "date",
        frequency: str | pd.Timedelta | None = None,
        validate: bool = True,
        require_complete_grid: bool = True,
        require_complete: bool | None = None,
        **kwargs: Any,
    ) -> GeoPanel:
        adapter = CallablePanelAdapter(
            fetcher=fetcher,
            geo_col=geo_col,
            time_col=time_col,
            kwargs=kwargs,
        )
        return cls.from_adapter(
            adapter,
            geos=geos,
            start=start,
            end=end,
            metrics=metrics,
            columns=columns,
            geo_col=geo_col,
            time_col=time_col,
            frequency=frequency,
            validate=validate,
            require_complete_grid=require_complete_grid,
            require_complete=require_complete,
        )

    @classmethod
    def from_adapter(
        cls,
        adapter: PanelAdapter,
        *,
        geos: Iterable[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        metrics: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
        geo_col: str = "geo_id",
        time_col: str = "date",
        frequency: str | pd.Timedelta | None = None,
        validate: bool = True,
        require_complete_grid: bool = True,
        require_complete: bool | None = None,
    ) -> GeoPanel:
        result = adapter.fetch(geos=geos, start=start, end=end, metrics=metrics, columns=columns)
        if isinstance(result, GeoPanel):
            frame = result.to_pandas()
            geo_col = result.geo_col
            time_col = result.time_col
            frequency = result.frequency
        else:
            frame = pd.DataFrame(result)
        return cls.from_dataframe(
            frame,
            geo_col=geo_col,
            time_col=time_col,
            frequency=frequency,
            validate=validate,
            require_complete_grid=require_complete_grid,
            require_complete=require_complete,
        )

    @staticmethod
    def _normalize_frame(df: pd.DataFrame, geo_col: str, time_col: str) -> pd.DataFrame:
        require_columns(df, [geo_col, time_col])
        frame = df.copy()
        frame[geo_col] = frame[geo_col].astype(str)
        frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce", utc=True).dt.tz_localize(
            None
        )
        if frame[time_col].isna().any():
            raise ValidationError(
                f"{time_col!r} contains null or unparseable timestamps.",
                remediation="Clean the time column before constructing a GeoPanel.",
            )
        if frame[geo_col].isna().any():
            raise ValidationError(
                f"{geo_col!r} contains null geography identifiers.",
                remediation="Drop or impute rows with missing market identifiers.",
            )
        return frame.sort_values([geo_col, time_col], kind="mergesort").reset_index(drop=True)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.to_pandas()

    @property
    def markets(self) -> list[str]:
        return sorted(self.df[self.geo_col].astype(str).unique().tolist())

    @property
    def dates(self) -> list[pd.Timestamp]:
        return list(pd.DatetimeIndex(self.df[self.time_col].drop_duplicates().sort_values()))

    @property
    def times(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.dates)

    @property
    def time_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        if self.df.empty:
            raise ValidationError(
                "Cannot compute a time range for an empty panel.",
                remediation="Fetch or construct a non-empty panel.",
            )
        return (self.df[self.time_col].min(), self.df[self.time_col].max())

    @property
    def metric_columns(self) -> list[str]:
        excluded = {self.geo_col, self.time_col}
        if self.market_metadata is not None:
            excluded.update(self.market_metadata.columns)
        return [
            col
            for col in self.df.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(self.df[col])
        ]

    def validate(
        self,
        *,
        required_columns: Iterable[str] | None = None,
        require_complete_grid: bool = True,
        require_complete: bool | None = None,
    ) -> PanelValidationResult:
        if require_complete is not None:
            require_complete_grid = require_complete
        result = validate_long_panel(
            self.df,
            geo_col=self.geo_col,
            time_col=self.time_col,
            required_columns=required_columns,
            frequency=self.frequency,
            require_complete_grid=require_complete_grid,
        )
        self._validation = result
        return result

    def complete_grid_diagnostics(self) -> CompleteGridDiagnostics:
        return complete_grid_diagnostics(
            self.df,
            geo_col=self.geo_col,
            time_col=self.time_col,
            frequency=self.frequency,
        )

    def slice(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        *,
        geos: Iterable[str] | None = None,
    ) -> GeoPanel:
        frame = self.df
        if start is not None:
            frame = frame.loc[frame[self.time_col] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame[self.time_col] <= pd.Timestamp(end)]
        if geos is not None:
            geo_set = {str(geo) for geo in geos}
            frame = frame.loc[frame[self.geo_col].isin(geo_set)]
        return GeoPanel.from_dataframe(
            frame,
            geo_col=self.geo_col,
            time_col=self.time_col,
            frequency=self.frequency,
            market_metadata=self.market_metadata,
            validate=True,
            require_complete_grid=False,
        )

    def metric_frame(
        self,
        metrics: Iterable[str],
        *,
        include_keys: bool = True,
        include_geo_time: bool | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        geos: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        metric_list = list(dict.fromkeys(metrics))
        missing = [metric for metric in metric_list if metric not in self.df.columns]
        if missing:
            raise ValidationError(
                f"panel missing metric columns: {missing}",
                remediation="Load the required metric columns before analysis.",
            )
        include_keys = include_keys if include_geo_time is None else include_geo_time
        panel = self.slice(start, end, geos=geos) if any([start, end, geos]) else self
        cols = [self.geo_col, self.time_col] if include_keys else []
        return panel.df[[*cols, *metric_list]].copy()

    def aggregate(
        self,
        columns: Iterable[str],
        *,
        geos: Iterable[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        by: str | list[str] | None = None,
        freq: str | pd.Timedelta | None = None,
        agg: str | Mapping[str, str] = "sum",
    ) -> pd.DataFrame:
        panel = self.slice(start, end, geos=geos) if any([start, end, geos]) else self
        column_list = list(columns)
        require_columns(panel.df, column_list, context="panel metrics")
        if freq is not None:
            grouping: list[Any] = []
            if by is None:
                grouping.append(panel.geo_col)
            elif isinstance(by, str):
                grouping.append(by)
            else:
                grouping.extend(by)
            grouping.append(pd.Grouper(key=panel.time_col, freq=freq))
            return (
                panel.df.groupby(grouping, as_index=False, observed=True, sort=True)[column_list]
                .agg(agg)
                .sort_values(
                    [key for key in [*grouping[:-1], panel.time_col] if isinstance(key, str)],
                    kind="mergesort",
                )
                .reset_index(drop=True)
            )
        if by is None:
            return panel.df[column_list].agg(agg).to_frame().T
        return panel.df.groupby(by, as_index=False, observed=True)[column_list].agg(agg)

    def aggregate_metrics(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.aggregate(*args, **kwargs)

    def attach_market_metadata(
        self,
        metadata: pd.DataFrame,
        *,
        geo_col: str | None = None,
        validate: bool = True,
    ) -> GeoPanel:
        metadata_geo_col = geo_col or self.geo_col
        require_columns(metadata, [metadata_geo_col], context="market metadata")
        meta = metadata.copy()
        if metadata_geo_col != self.geo_col:
            meta = meta.rename(columns={metadata_geo_col: self.geo_col})
        if validate:
            unknown = set(meta[self.geo_col].astype(str)) - set(self.markets)
            if unknown:
                raise ValidationError(
                    f"Market metadata contains {len(unknown)} unknown market(s).",
                    remediation="Filter metadata to panel markets or disable validation.",
                )
        merged = self.df.merge(meta, on=self.geo_col, how="left")
        return GeoPanel.from_dataframe(
            merged,
            geo_col=self.geo_col,
            time_col=self.time_col,
            frequency=self.frequency,
            market_metadata=meta,
            validate=True,
            require_complete_grid=False,
        )

    def require_metric_columns(self, metrics: Iterable[Any]) -> None:
        columns: list[str] = []
        for metric in metrics:
            required = getattr(metric, "required_columns", None)
            columns.extend(list(required() if callable(required) else required or []))
        require_columns(self.df, columns, context="panel metrics")

    def to_pandas(self, *, copy: bool = True) -> pd.DataFrame:
        return self.df.copy() if copy else self.df

    def to_polars(self) -> Any:
        try:
            import polars as pl  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError("GeoPanel.to_polars requires optional dependency 'polars'.") from exc
        return pl.from_pandas(self.df)

    def to_parquet(self, path: str | Path, **write_kwargs: Any) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(out, index=False, **write_kwargs)
        return out

    def to_duckdb(
        self,
        database_or_connection: str | Path | Any,
        table: str = "geo_panel",
        *,
        replace: bool = True,
        table_name: str | None = None,
    ) -> str | Path | Any:
        import duckdb

        target_table = table_name or table
        mode = "create or replace" if replace else "create"
        if hasattr(database_or_connection, "execute") and hasattr(
            database_or_connection,
            "register",
        ):
            con = database_or_connection
            con.register("_fieldtrial_panel", self.df)
            try:
                con.execute(
                    f"{mode} table {quote_sql_qualified_name(target_table)} "
                    "as select * from _fieldtrial_panel"
                )
            finally:
                con.unregister("_fieldtrial_panel")
            return con

        database = Path(database_or_connection)
        database.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(database)) as con:
            con.register("_fieldtrial_panel", self.df)
            con.execute(
                f"{mode} table {quote_sql_qualified_name(target_table)} "
                "as select * from _fieldtrial_panel"
            )
        return database

    def __len__(self) -> int:
        return len(self.df)


__all__ = ["GeoPanel"]
