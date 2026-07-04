"""Ratio metric utilities including delta-method diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats

from fieldtrial.data.validation import require_columns
from fieldtrial.metrics.base import MetricSpec


@dataclass(frozen=True)
class RatioDeltaResult:
    treatment_ratio: float
    control_ratio: float
    difference: float
    relative_lift: float
    standard_error: float
    interval: tuple[float, float]
    p_value: float
    diagnostics: dict[str, float]

    @property
    def variance(self) -> float:
        return float(self.standard_error**2)

    @property
    def ci_low(self) -> float:
        return self.interval[0]

    @property
    def ci_high(self) -> float:
        return self.interval[1]

    def to_dict(self) -> dict[str, object]:
        return {
            "treatment_ratio": self.treatment_ratio,
            "control_ratio": self.control_ratio,
            "difference": self.difference,
            "relative_lift": self.relative_lift,
            "standard_error": self.standard_error,
            "variance": self.variance,
            "interval": self.interval,
            "p_value": self.p_value,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class RatioMetric(MetricSpec):
    numerator: str | None = None
    denominator: str | None = None
    denominator_min: float = 1e-12
    metric_type: str = "ratio"
    zero_denominator: str = "nan"

    def __post_init__(self) -> None:
        if self.numerator is None or self.denominator is None:
            raise ValueError("RatioMetric requires numerator and denominator")

    @property
    def required_columns(self) -> list[str]:
        return [str(self.numerator), str(self.denominator)]

    def compute_series(self, df: pd.DataFrame) -> pd.Series:
        require_columns(df, self.required_columns, context=f"metric {self.name!r}")
        numerator = pd.to_numeric(df[str(self.numerator)], errors="coerce")
        denominator = pd.to_numeric(df[str(self.denominator)], errors="coerce")
        bad = denominator <= 0
        if bad.any():
            message = (
                f"RatioMetric {self.name!r} encountered {int(bad.sum())} "
                "non-positive denominator row(s)."
            )
            if self.zero_denominator == "raise":
                raise ZeroDivisionError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return numerator.divide(denominator.where(~bad))

    def aggregate(self, df: pd.DataFrame) -> float:
        return group_ratio(
            df,
            numerator=str(self.numerator),
            denominator=str(self.denominator),
            denominator_min=self.denominator_min,
        )

    def inject_lift(
        self,
        df: pd.DataFrame,
        lift: float,
        *,
        relative: bool = True,
        affect_denominator: bool = False,
        target_mask: pd.Series | None = None,
    ) -> pd.DataFrame:
        out = df.copy()
        mask = target_mask if target_mask is not None else pd.Series(True, index=out.index)
        out[str(self.numerator)] = pd.to_numeric(
            out[str(self.numerator)],
            errors="coerce",
        ).astype(float)
        out[str(self.denominator)] = pd.to_numeric(
            out[str(self.denominator)],
            errors="coerce",
        ).astype(float)
        if relative:
            out.loc[mask, str(self.numerator)] = out.loc[mask, str(self.numerator)] * (1 + lift)
            if affect_denominator:
                out.loc[mask, str(self.denominator)] = out.loc[mask, str(self.denominator)] * (
                    1 + lift
                )
        else:
            out.loc[mask, str(self.numerator)] = out.loc[mask, str(self.numerator)] + lift
            if affect_denominator:
                out.loc[mask, str(self.denominator)] = out.loc[mask, str(self.denominator)] + lift
        return out

    def linearized_residuals(self, df: pd.DataFrame, ratio: float | None = None) -> pd.Series:
        if ratio is None:
            ratio = self.aggregate(df)
        require_columns(df, self.required_columns, context=f"metric {self.name!r}")
        return df[str(self.numerator)] - ratio * df[str(self.denominator)]

    def ratio_variance(self, df: pd.DataFrame, *, cluster_col: str | None = None) -> float:
        ratio = self.aggregate(df)
        variance, _ = _ratio_variance(
            df,
            numerator=str(self.numerator),
            denominator=str(self.denominator),
            ratio=ratio,
            cluster_col=cluster_col,
        )
        return variance

    def difference(
        self,
        treatment: pd.DataFrame,
        control: pd.DataFrame,
        *,
        alpha: float = 0.05,
        cluster_col: str | None = None,
    ) -> RatioDeltaResult:
        return delta_method_difference(
            treatment,
            control,
            numerator=str(self.numerator),
            denominator=str(self.denominator),
            alpha=alpha,
            denominator_min=self.denominator_min,
            cluster_col=cluster_col,
        )

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload.update(
            {
                "numerator": self.numerator,
                "denominator": self.denominator,
                "denominator_min": self.denominator_min,
                "zero_denominator": self.zero_denominator,
            }
        )
        return payload


def group_ratio(
    df: pd.DataFrame,
    *,
    numerator: str,
    denominator: str,
    denominator_min: float = 1e-12,
) -> float:
    require_columns(df, [numerator, denominator], context="ratio metric")
    den = float(df[denominator].sum())
    if den < denominator_min:
        raise ZeroDivisionError(
            f"ratio denominator {denominator!r} below minimum {denominator_min}"
        )
    if (df[denominator] <= 0).any():
        warnings.warn(
            f"ratio denominator {denominator!r} contains non-positive row values",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(df[numerator].sum() / den)


def _ratio_variance(
    df: pd.DataFrame,
    *,
    numerator: str,
    denominator: str,
    ratio: float,
    cluster_col: str | None,
) -> tuple[float, int]:
    if cluster_col is not None:
        require_columns(df, [cluster_col], context="ratio variance clustering")
        grouped = df.assign(_lin=df[numerator] - ratio * df[denominator]).groupby(cluster_col)
        residual = grouped["_lin"].sum().to_numpy(dtype=float)
        den = grouped[denominator].sum().to_numpy(dtype=float)
        n = len(residual)
        if n <= 1:
            return float("nan"), n
        variance = float(np.var(residual, ddof=1) / n / (np.mean(den) ** 2))
        return variance, n
    residual = (df[numerator] - ratio * df[denominator]).to_numpy(dtype=float)
    den_sum = float(df[denominator].sum())
    n = len(residual)
    if n <= 1 or den_sum == 0:
        return float("nan"), n
    variance = float(n * np.var(residual, ddof=1) / (den_sum**2))
    return variance, n


def delta_method_difference(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    *,
    numerator: str,
    denominator: str,
    alpha: float = 0.05,
    denominator_min: float = 1e-12,
    cluster_col: str | None = None,
) -> RatioDeltaResult:
    tr = group_ratio(
        treatment,
        numerator=numerator,
        denominator=denominator,
        denominator_min=denominator_min,
    )
    cr = group_ratio(
        control,
        numerator=numerator,
        denominator=denominator,
        denominator_min=denominator_min,
    )
    vt, nt = _ratio_variance(
        treatment,
        numerator=numerator,
        denominator=denominator,
        ratio=tr,
        cluster_col=cluster_col,
    )
    vc, nc = _ratio_variance(
        control,
        numerator=numerator,
        denominator=denominator,
        ratio=cr,
        cluster_col=cluster_col,
    )
    if nt <= 1 or nc <= 1 or not np.isfinite(vt) or not np.isfinite(vc):
        unit_label = "clusters" if cluster_col else "rows"
        raise ValueError(
            f"Delta-method ratio difference requires at least two finite {unit_label} in each arm."
        )
    se = sqrt(max(vt + vc, 0.0))
    diff = tr - cr
    df_terms = []
    if nt > 1 and vt > 0:
        df_terms.append((vt**2) / (nt - 1))
    if nc > 1 and vc > 0:
        df_terms.append((vc**2) / (nc - 1))
    df = (vt + vc) ** 2 / sum(df_terms) if df_terms else max(nt + nc - 2, 1)
    critical = stats.t.ppf(1 - alpha / 2, df=df)
    interval = (diff - critical * se, diff + critical * se)
    if se > 0:
        p_value = float(2 * (1 - stats.t.cdf(abs(diff / se), df=df)))
    else:
        p_value = 1.0 if diff == 0 else 0.0
    rel = diff / cr if cr else float("nan")
    return RatioDeltaResult(
        treatment_ratio=tr,
        control_ratio=cr,
        difference=diff,
        relative_lift=float(rel),
        standard_error=float(se),
        interval=(float(interval[0]), float(interval[1])),
        p_value=p_value,
        diagnostics={
            "treatment_denominator": float(treatment[denominator].sum()),
            "control_denominator": float(control[denominator].sum()),
            "treatment_units": float(nt),
            "control_units": float(nc),
            "degrees_of_freedom": float(df),
            "reference_distribution": "welch_satterthwaite_t",
        },
    )
