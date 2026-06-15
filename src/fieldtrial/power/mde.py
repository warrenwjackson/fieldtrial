"""MDE approximations used by planning workflows."""

from __future__ import annotations

import math

import pandas as pd
from scipy import optimize, stats

from fieldtrial.metrics.ratio import RatioMetric


def approximate_count_mde(
    treatment: pd.Series,
    control: pd.Series,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Approximate relative MDE for aggregate count-like outcomes."""

    _validate_power_inputs(alpha=alpha, power=power)
    t = treatment.astype(float)
    c = control.astype(float)
    if not t.index.equals(c.index):
        raise ValueError("treatment and control series must have identical indexes")
    if len(t) < 2:
        raise ValueError("MDE requires at least two paired observations")
    if not t.notna().all() or not c.notna().all():
        raise ValueError("MDE inputs must not contain missing values")
    if not t.map(math.isfinite).all() or not c.map(math.isfinite).all():
        raise ValueError("MDE inputs must be finite")
    baseline = float(t.mean())
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("relative MDE requires a positive treatment baseline")
    noise = float((t - c).std(ddof=1))
    if not math.isfinite(noise):
        raise ValueError("MDE noise estimate must be finite")
    df = len(t) - 1
    ncp = _two_sided_noncentral_t_ncp(alpha=alpha, power=power, df=df)
    standard_error = noise / math.sqrt(len(t))
    return float(max(0.0, ncp * standard_error / baseline))


def ratio_delta_mde(
    df: pd.DataFrame,
    metric: RatioMetric,
    *,
    treatment_geos: list[str],
    control_geos: list[str],
    geo_col: str = "geo_id",
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    _validate_power_inputs(alpha=alpha, power=power)
    treatment = df[df[geo_col].isin(treatment_geos)]
    control = df[df[geo_col].isin(control_geos)]
    result = metric.difference(treatment, control, alpha=alpha, cluster_col=geo_col)
    default_df = max(len(treatment_geos) + len(control_geos) - 2, 1)
    df = float(
        result.diagnostics.get(
            "degrees_of_freedom",
            default_df,
        )
    )
    ncp = _two_sided_noncentral_t_ncp(alpha=alpha, power=power, df=df)
    if not math.isfinite(result.standard_error):
        raise ValueError("ratio MDE standard error must be finite")
    baseline = abs(float(result.control_ratio))
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("relative ratio MDE requires a non-zero control ratio")
    return float(ncp * result.standard_error / baseline)


def _validate_power_inputs(*, alpha: float, power: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 < power < 1:
        raise ValueError("power must be between 0 and 1")


def _two_sided_noncentral_t_ncp(*, alpha: float, power: float, df: float) -> float:
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))

    def achieved_power(ncp: float) -> float:
        return float(
            1.0 - stats.nct.cdf(critical, df=df, nc=ncp) + stats.nct.cdf(-critical, df=df, nc=ncp)
        )

    if power <= achieved_power(0.0):
        return 0.0
    upper = 1.0
    while achieved_power(upper) < power:
        upper *= 2.0
        if upper > 1e6:
            raise ValueError("could not bracket noncentral-t MDE solution")
    return float(
        optimize.brentq(
            lambda ncp: achieved_power(ncp) - power,
            0.0,
            upper,
            xtol=1e-10,
            rtol=1e-10,
        )
    )
