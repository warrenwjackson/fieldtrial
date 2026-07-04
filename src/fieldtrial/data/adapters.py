"""Adapters for fetching panel data from existing systems."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Any, Protocol

import pandas as pd

from fieldtrial.exceptions import ValidationError


def quote_sql_identifier(identifier: str) -> str:
    """Quote one SQL identifier part using ANSI double-quote escaping."""

    value = str(identifier).strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        value = value[1:-1].replace('""', '"')
    if not value or "\x00" in value or ";" in value:
        raise ValidationError(
            f"Unsafe SQL identifier: {identifier!r}",
            remediation="Pass a table or column name, not a SQL expression.",
        )
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def quote_sql_qualified_name(identifier: str) -> str:
    """Quote a possibly schema-qualified table/view name."""

    parts = [part.strip() for part in str(identifier).split(".")]
    if not parts or any(not part for part in parts):
        raise ValidationError(
            f"Unsafe SQL identifier: {identifier!r}",
            remediation="Pass a table or schema.table name.",
        )
    return ".".join(quote_sql_identifier(part) for part in parts)


def validate_sql_filter_clause(clause: str) -> str:
    """Reject multi-statement/comment WHERE fragments for trusted DuckDB loading.

    This helper is a guardrail for convenience filters, not a SQL-injection sanitizer.
    Use ``GeoPanel.from_query(..., params=...)`` for untrusted values.
    """

    value = str(clause).strip()
    lowered = value.lower()
    if not value or "\x00" in value or ";" in value or "--" in value or "/*" in lowered:
        raise ValidationError(
            "Unsafe SQL WHERE clause.",
            remediation=(
                "Pass a single predicate expression without statement separators or comments; "
                "use GeoPanel.from_query(..., params=...) for untrusted values."
            ),
        )
    return value


class PanelAdapter(Protocol):
    """Protocol for objects that fetch long-format panel data."""

    def fetch(
        self,
        *,
        geos: Iterable[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        metrics: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame: ...


def _coerce_to_dataframe(result: Any) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    if hasattr(result, "df"):
        return result.df()
    if hasattr(result, "fetchdf"):
        return result.fetchdf()
    return pd.DataFrame(result)


def _filter_panel_frame(
    frame: pd.DataFrame,
    *,
    geo_col: str,
    time_col: str,
    geos: Iterable[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    metrics: Iterable[str] | None = None,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True).dt.tz_localize(
            None
        )
    if start is not None:
        out = out.loc[out[time_col] >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out[time_col] <= pd.Timestamp(end)]
    if geos is not None:
        out = out.loc[out[geo_col].astype(str).isin({str(geo) for geo in geos})]
    requested = list(dict.fromkeys([*(metrics or []), *(columns or [])]))
    if requested:
        keep = list(dict.fromkeys([geo_col, time_col, *requested]))
        missing = [column for column in keep if column not in out.columns]
        if missing:
            raise ValidationError(
                f"Fetched panel is missing requested column(s): {', '.join(missing)}",
                remediation="Update the query, fetch callable, or metric definitions.",
            )
        out = out.loc[:, keep]
    return out.reset_index(drop=True)


@dataclass
class DataFramePanelAdapter:
    """Adapter over an in-memory DataFrame."""

    frame: pd.DataFrame
    geo_col: str = "geo_id"
    time_col: str = "date"

    def fetch(
        self,
        *,
        geos: Iterable[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        metrics: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        return _filter_panel_frame(
            self.frame,
            geo_col=self.geo_col,
            time_col=self.time_col,
            geos=geos,
            start=start,
            end=end,
            metrics=metrics,
            columns=columns,
        )


@dataclass
class CallablePanelAdapter:
    """Adapter for user-provided fetch functions."""

    fetcher: Callable[..., Any]
    geo_col: str = "geo_id"
    time_col: str = "date"
    frequency: str | pd.Timedelta | None = None
    extra_kwargs: Mapping[str, Any] | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def fetch(
        self,
        *,
        geos: Iterable[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        metrics: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        requested = list(dict.fromkeys([*(metrics or []), *(columns or [])]))
        call_kwargs = {
            "geos": geos,
            "start": start,
            "end": end,
            "metrics": requested or metrics,
            "columns": requested or columns,
            **dict(self.extra_kwargs or {}),
            **dict(self.kwargs),
        }
        result = self.fetcher(**self._accepted_kwargs(call_kwargs))
        frame = _coerce_to_dataframe(result)
        filtered = _filter_panel_frame(
            frame,
            geo_col=self.geo_col,
            time_col=self.time_col,
            geos=geos,
            start=start,
            end=end,
            metrics=metrics,
            columns=columns,
        )
        from fieldtrial.data.panel import GeoPanel

        return GeoPanel.from_dataframe(
            filtered,
            geo_col=self.geo_col,
            time_col=self.time_col,
            frequency=self.frequency,
            require_complete_grid=False,
        )

    def _accepted_kwargs(self, call_kwargs: Mapping[str, Any]) -> dict[str, Any]:
        sig = signature(self.fetcher)
        if any(param.kind == Parameter.VAR_KEYWORD for param in sig.parameters.values()):
            return dict(call_kwargs)
        return {key: value for key, value in call_kwargs.items() if key in sig.parameters}


@dataclass
class SQLQueryPanelAdapter:
    """Adapter for DuckDB, DB-API, and SQLAlchemy-style query execution."""

    connection: Any
    query: str
    geo_col: str = "geo_id"
    time_col: str = "date"
    params: Mapping[str, Any] | Sequence[Any] | None = None
    frequency: str | pd.Timedelta | None = None

    def fetch(
        self,
        *,
        geos: Iterable[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        metrics: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        frame = read_sql_query(self.connection, self.query, params=self.params)
        return _filter_panel_frame(
            frame,
            geo_col=self.geo_col,
            time_col=self.time_col,
            geos=geos,
            start=start,
            end=end,
            metrics=metrics,
            columns=columns,
        )


QueryPanelAdapter = SQLQueryPanelAdapter


@dataclass
class DuckDBTableAdapter:
    """Adapter for a table or view registered in DuckDB."""

    connection: Any
    table_name: str
    geo_col: str = "geo_id"
    time_col: str = "date"

    def fetch(
        self,
        *,
        geos: Iterable[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        metrics: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        requested = list(dict.fromkeys([*(metrics or []), *(columns or [])]))
        select = "*"
        if requested:
            selected_columns = [self.geo_col, self.time_col, *requested]
            select = ", ".join(quote_sql_identifier(column) for column in selected_columns)
        query = f"select {select} from {quote_sql_qualified_name(self.table_name)}"
        return SQLQueryPanelAdapter(
            self.connection,
            query,
            geo_col=self.geo_col,
            time_col=self.time_col,
        ).fetch(geos=geos, start=start, end=end, metrics=metrics, columns=columns)


def read_sql_query(
    connection: Any,
    query: str,
    *,
    params: Mapping[str, Any] | Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Execute a query and return a DataFrame across common connection types."""

    if hasattr(connection, "sql"):
        try:
            relation = (
                connection.sql(query, params=params)
                if params is not None
                else connection.sql(query)
            )
        except Exception as exc:
            raise ValidationError(
                f"Unable to execute panel query: {exc}",
                remediation=(
                    "Check the SQL and parameter style for this connection. FieldTrial does not "
                    "retry parameterized queries without their filters."
                ),
            ) from exc
        return _coerce_to_dataframe(relation)

    if hasattr(connection, "execute"):
        try:
            cursor = (
                connection.execute(query, params)
                if params is not None
                else connection.execute(query)
            )
        except Exception as exc:
            raise ValidationError(
                f"Unable to execute panel query: {exc}",
                remediation=(
                    "Check the SQL and parameter style for this connection. FieldTrial does not "
                    "retry parameterized queries without their filters."
                ),
            ) from exc
        if isinstance(cursor, pd.DataFrame):
            return cursor.copy()
        if hasattr(cursor, "df"):
            return cursor.df()
        if hasattr(cursor, "fetchdf"):
            return cursor.fetchdf()
        if hasattr(cursor, "fetchall"):
            rows = cursor.fetchall()
            columns = [desc[0] for desc in getattr(cursor, "description", [])]
            return pd.DataFrame(rows, columns=columns or None)

    try:
        return pd.read_sql_query(query, connection, params=params)
    except Exception as exc:  # pragma: no cover - backend exceptions vary
        raise ValidationError(
            f"Unable to execute panel query: {exc}",
            remediation="Pass a DuckDB, DB-API, or SQLAlchemy connection and a valid SQL query.",
        ) from exc


__all__ = [
    "CallablePanelAdapter",
    "DataFramePanelAdapter",
    "DuckDBTableAdapter",
    "PanelAdapter",
    "QueryPanelAdapter",
    "SQLQueryPanelAdapter",
    "quote_sql_identifier",
    "quote_sql_qualified_name",
    "read_sql_query",
    "validate_sql_filter_clause",
]
