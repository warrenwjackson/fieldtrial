"""Cross-test covariance primitives for experiment roadmaps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from fieldtrial.portfolio._utils import (
    as_date,
    as_tuple,
    finite_float,
    inclusive_day_count,
    jsonable,
    overlap_day_count,
    required_attr,
    safe_ratio,
)


@dataclass(frozen=True)
class PortfolioEstimate:
    """One test-metric estimate with enough metadata to reason about dependence."""

    test_id: str
    metric: str
    estimate: float
    standard_error: float | None = None
    variance: float | None = None
    p_value: float | None = None
    interval: tuple[float, float] | None = None
    treatment_markets: tuple[str, ...] | list[str] = field(default_factory=tuple)
    control_markets: tuple[str, ...] | list[str] = field(default_factory=tuple)
    start_date: Any | None = None
    end_date: Any | None = None
    method_family: str = "unknown"
    estimator_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        estimate = finite_float(self.estimate)
        if estimate is None:
            raise ValueError("estimate must be finite")
        interval = self.interval
        if interval is not None:
            lower, upper = interval
            lower_float = finite_float(lower)
            upper_float = finite_float(upper)
            if lower_float is None or upper_float is None or upper_float < lower_float:
                raise ValueError("interval must be a finite (lower, upper) tuple")
            interval = (lower_float, upper_float)
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "standard_error", finite_float(self.standard_error))
        object.__setattr__(self, "variance", finite_float(self.variance))
        object.__setattr__(self, "p_value", finite_float(self.p_value))
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "treatment_markets", as_tuple(self.treatment_markets))
        object.__setattr__(self, "control_markets", as_tuple(self.control_markets))
        object.__setattr__(self, "start_date", as_date(self.start_date))
        object.__setattr__(self, "end_date", as_date(self.end_date))

    @property
    def key(self) -> str:
        return f"{self.test_id}:{self.metric}"

    @property
    def all_markets(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.treatment_markets, *self.control_markets)))

    @property
    def duration_days(self) -> int:
        return inclusive_day_count(self.start_date, self.end_date)

    def resolved_variance(self, *, confidence: float = 0.95) -> tuple[float | None, str]:
        if self.variance is not None and self.variance > 0:
            return self.variance, "variance"
        if self.standard_error is not None and self.standard_error > 0:
            return self.standard_error**2, "standard_error"
        if self.interval is not None:
            df = finite_float(self.metadata.get("degrees_of_freedom"))
            critical = (
                stats.t.ppf(0.5 + confidence / 2.0, df=df)
                if df is not None and df > 0
                else stats.norm.ppf(0.5 + confidence / 2.0)
            )
            standard_error = (self.interval[1] - self.interval[0]) / (2.0 * critical)
            if standard_error > 0:
                return standard_error**2, "interval_t" if df is not None else "interval"
        return None, "missing"

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_estimator_result(
        cls,
        result: Any,
        design: Any,
        *,
        test_id: str | None = None,
        effect_scale: str = "estimate",
    ) -> PortfolioEstimate:
        """Build a portfolio estimate from existing estimator/design-like objects."""

        estimate_value = getattr(result, effect_scale)
        metadata = getattr(result, "method_metadata", None)
        method_family = "unknown"
        if metadata is not None:
            method_family = str(
                getattr(metadata, "independent_family", None)
                or getattr(metadata, "family", None)
                or "unknown"
            )
        primary = _primary_inference_payload(result)
        return cls(
            test_id=test_id or getattr(design, "experiment_id", getattr(result, "test_id", "")),
            metric=result.metric,
            estimate=estimate_value,
            standard_error=primary.get("standard_error", getattr(result, "standard_error", None)),
            p_value=primary.get("p_value", getattr(result, "p_value", None)),
            interval=primary.get("interval", getattr(result, "interval", None)),
            treatment_markets=getattr(design, "treatment_geos", ()),
            control_markets=getattr(design, "control_geos", ()),
            start_date=getattr(design, "start_date", None),
            end_date=getattr(design, "end_date", None),
            method_family=method_family,
            estimator_name=getattr(result, "estimator_name", None),
            metadata=primary.get("metadata", {}),
        )


@dataclass(frozen=True)
class PortfolioCovariance:
    """Covariance and correlation artifact for a set of portfolio estimates."""

    estimate_keys: tuple[str, ...]
    covariance: tuple[tuple[float, ...], ...]
    correlation: tuple[tuple[float, ...], ...]
    drivers: dict[str, dict[str, Any]]
    method: str
    warnings: tuple[str, ...] = ()
    artifact_version: str = "fieldtrial.portfolio.covariance.v1"

    def covariance_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.covariance, index=self.estimate_keys, columns=self.estimate_keys)

    def correlation_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.correlation,
            index=self.estimate_keys,
            columns=self.estimate_keys,
        )

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


def estimate_cross_test_covariance(
    estimates: Sequence[PortfolioEstimate | Mapping[str, Any] | Any],
    *,
    draws: pd.DataFrame | Mapping[str, Sequence[float]] | None = None,
    confidence: float = 0.95,
    max_proxy_correlation: float = 0.95,
    minimum_variance: float = 1e-12,
) -> PortfolioCovariance:
    """Estimate covariance across test-metric results.

    Empirical draws are used whenever a pair has aligned draw columns. Remaining
    pairs use an auditable proxy from shared controls, repeated markets, and
    calendar overlap so portfolio artifacts do not silently assume independence.
    """

    normalized = [_coerce_estimate(item) for item in estimates]
    _validate_unique_keys(normalized)
    keys = tuple(item.key for item in normalized)
    if not normalized:
        return PortfolioCovariance(
            estimate_keys=(),
            covariance=(),
            correlation=(),
            drivers={},
            method="empty",
        )

    draw_frame = _normalize_draws(draws)
    variances: list[float] = []
    variance_sources: list[str] = []
    warnings: list[str] = []
    for item in normalized:
        variance, source = item.resolved_variance(confidence=confidence)
        if draw_frame is not None and item.key in draw_frame:
            draw_variance = finite_float(draw_frame[item.key].dropna().var(ddof=1))
            if draw_variance is not None and draw_variance > 0:
                variance = draw_variance
                source = "draws"
        if variance is None or variance <= 0:
            variance = minimum_variance
            source = "minimum_variance"
            warnings.append(
                f"{item.key} had no usable variance, standard error, interval, or draws."
            )
        variances.append(float(variance))
        variance_sources.append(source)

    covariance = np.zeros((len(normalized), len(normalized)), dtype=float)
    drivers: dict[str, dict[str, Any]] = {}
    for index, variance in enumerate(variances):
        covariance[index, index] = variance
        key = _pair_key(keys[index], keys[index])
        drivers[key] = {
            "source": variance_sources[index],
            "variance": variance,
            "test_id": normalized[index].test_id,
            "metric": normalized[index].metric,
        }

    for left_index in range(len(normalized)):
        for right_index in range(left_index + 1, len(normalized)):
            left = normalized[left_index]
            right = normalized[right_index]
            pair_key = _pair_key(left.key, right.key)
            draw_covariance = _pair_draw_covariance(draw_frame, left.key, right.key)
            if draw_covariance is not None:
                covariance_value, draw_count = draw_covariance
                source = "draws"
                driver = _overlap_driver(left, right)
                driver.update({"source": source, "draw_count": draw_count})
            else:
                driver = _overlap_driver(left, right)
                correlation = float(
                    np.clip(
                        driver["proxy_correlation"],
                        -max_proxy_correlation,
                        max_proxy_correlation,
                    )
                )
                variance_product = variances[left_index] * variances[right_index]
                covariance_value = correlation * float(np.sqrt(variance_product))
                source = "overlap_proxy"
                driver.update({"source": source, "draw_count": 0})
            covariance[left_index, right_index] = covariance_value
            covariance[right_index, left_index] = covariance_value
            drivers[pair_key] = driver

    correlation = _covariance_to_correlation(covariance)
    projected = _project_correlation_if_needed(correlation)
    if projected is not correlation:
        warnings.append("Correlation matrix was projected to the nearest PSD approximation.")
        covariance = projected * np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        correlation = projected

    method = "draws_and_overlap_proxy" if draw_frame is not None else "overlap_proxy"
    return PortfolioCovariance(
        estimate_keys=keys,
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        correlation=tuple(tuple(float(value) for value in row) for row in correlation),
        drivers=drivers,
        method=method,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def covariance_clusters(
    covariance: PortfolioCovariance,
    *,
    threshold: float = 0.35,
) -> list[dict[str, Any]]:
    """Return connected covariance clusters above an absolute correlation threshold."""

    keys = list(covariance.estimate_keys)
    if not keys:
        return []
    corr = np.asarray(covariance.correlation, dtype=float)
    adjacency = {key: set[str]() for key in keys}
    for left_index, left_key in enumerate(keys):
        for right_index in range(left_index + 1, len(keys)):
            value = abs(float(corr[left_index, right_index]))
            if value >= threshold:
                right_key = keys[right_index]
                adjacency[left_key].add(right_key)
                adjacency[right_key].add(left_key)

    clusters: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen or not adjacency[key]:
            continue
        stack = [key]
        members: set[str] = set()
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            stack.extend(sorted(adjacency[current] - members))
        seen.update(members)
        indices = [keys.index(member) for member in sorted(members)]
        max_abs = 0.0
        for left_pos, left_index in enumerate(indices):
            for right_index in indices[left_pos + 1 :]:
                max_abs = max(max_abs, abs(float(corr[left_index, right_index])))
        clusters.append(
            {
                "members": sorted(members),
                "tests": sorted({member.split(":", 1)[0] for member in members}),
                "max_abs_correlation": max_abs,
                "threshold": threshold,
            }
        )
    return clusters


def _coerce_estimate(item: PortfolioEstimate | Mapping[str, Any] | Any) -> PortfolioEstimate:
    if isinstance(item, PortfolioEstimate):
        return item
    if isinstance(item, Mapping):
        return PortfolioEstimate(**dict(item))
    return PortfolioEstimate(
        test_id=required_attr(item, "test_id"),
        metric=required_attr(item, "metric"),
        estimate=required_attr(item, "estimate"),
        standard_error=getattr(item, "standard_error", None),
        variance=getattr(item, "variance", None),
        p_value=getattr(item, "p_value", None),
        interval=getattr(item, "interval", None),
        treatment_markets=getattr(item, "treatment_markets", ()),
        control_markets=getattr(item, "control_markets", ()),
        start_date=getattr(item, "start_date", None),
        end_date=getattr(item, "end_date", None),
        method_family=getattr(item, "method_family", "unknown"),
        estimator_name=getattr(item, "estimator_name", None),
        metadata=getattr(item, "metadata", {}),
    )


def _validate_unique_keys(estimates: Sequence[PortfolioEstimate]) -> None:
    keys = [item.key for item in estimates]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate portfolio estimate keys: {duplicates}")


def _normalize_draws(
    draws: pd.DataFrame | Mapping[str, Sequence[float]] | None,
) -> pd.DataFrame | None:
    if draws is None:
        return None
    frame = draws.copy() if isinstance(draws, pd.DataFrame) else pd.DataFrame(dict(draws))
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            ":".join(str(part) for part in column if part != "") for column in frame.columns
        ]
    else:
        frame.columns = [str(column) for column in frame.columns]
    return frame.apply(pd.to_numeric, errors="coerce")


def _pair_draw_covariance(
    draw_frame: pd.DataFrame | None,
    left_key: str,
    right_key: str,
) -> tuple[float, int] | None:
    if draw_frame is None or left_key not in draw_frame or right_key not in draw_frame:
        return None
    aligned = draw_frame[[left_key, right_key]].dropna()
    if len(aligned) < 2:
        return None
    covariance = finite_float(aligned.cov(ddof=1).iloc[0, 1])
    if covariance is None:
        return None
    return covariance, int(len(aligned))


def _primary_inference_payload(result: Any) -> dict[str, Any]:
    inference_results = list(getattr(result, "inference_results", []) or [])
    selected = next(
        (
            inference
            for inference in inference_results
            if getattr(inference, "diagnostics", {}).get("selected_as_primary")
        ),
        None,
    )
    if selected is None and inference_results:
        selected = inference_results[0]
    if selected is None:
        return {}
    metadata = {
        "inference_method": getattr(selected, "method", None),
        "interval_type": getattr(selected, "interval_type", None),
    }
    diagnostics = getattr(selected, "diagnostics", {}) or {}
    if diagnostics.get("degrees_of_freedom") is not None:
        metadata["degrees_of_freedom"] = diagnostics["degrees_of_freedom"]
    return {
        "standard_error": getattr(selected, "standard_error", None),
        "p_value": getattr(selected, "p_value", None),
        "interval": getattr(selected, "interval", None),
        "metadata": metadata,
    }


def _overlap_driver(left: PortfolioEstimate, right: PortfolioEstimate) -> dict[str, Any]:
    shared_controls = sorted(set(left.control_markets) & set(right.control_markets))
    shared_treatment = sorted(set(left.treatment_markets) & set(right.treatment_markets))
    repeated_markets = sorted(set(left.all_markets) & set(right.all_markets))
    overlap_days = overlap_day_count(
        left.start_date,
        left.end_date,
        right.start_date,
        right.end_date,
    )
    shortest_duration = min(left.duration_days, right.duration_days)
    role_alignment = _signed_role_alignment(left, right)
    same_role_markets, opposite_role_markets = _role_overlap_sets(left, right)
    control_overlap = safe_ratio(
        len(shared_controls),
        min(len(left.control_markets), len(right.control_markets)),
    )
    market_overlap = safe_ratio(
        len(repeated_markets),
        min(len(left.all_markets), len(right.all_markets)),
    )
    calendar_overlap = safe_ratio(overlap_days, shortest_duration)
    same_method_family = (
        left.method_family == right.method_family and left.method_family != "unknown"
    )
    dependence_scale = market_overlap * (0.75 * calendar_overlap + 0.25)
    if same_method_family and dependence_scale > 0:
        dependence_scale = min(1.0, dependence_scale + 0.05)
    proxy_correlation = role_alignment * dependence_scale
    return {
        "shared_control_markets": shared_controls,
        "shared_treatment_markets": shared_treatment,
        "repeated_markets": repeated_markets,
        "same_role_markets": same_role_markets,
        "opposite_role_markets": opposite_role_markets,
        "calendar_overlap_days": overlap_days,
        "calendar_overlap_fraction": calendar_overlap,
        "control_overlap_fraction": control_overlap,
        "market_overlap_fraction": market_overlap,
        "signed_role_alignment": role_alignment,
        "same_method_family": same_method_family,
        "proxy_correlation": float(np.clip(proxy_correlation, -0.95, 0.95)),
        "left_method_family": left.method_family,
        "right_method_family": right.method_family,
    }


def _role_weights(estimate: PortfolioEstimate) -> dict[str, float]:
    weights: dict[str, float] = {}
    treatment = tuple(dict.fromkeys(str(market) for market in estimate.treatment_markets))
    control = tuple(dict.fromkeys(str(market) for market in estimate.control_markets))
    if treatment:
        treatment_weight = 1.0 / len(treatment)
        for market in treatment:
            weights[market] = weights.get(market, 0.0) + treatment_weight
    if control:
        control_weight = -1.0 / len(control)
        for market in control:
            weights[market] = weights.get(market, 0.0) + control_weight
    return weights


def _signed_role_alignment(left: PortfolioEstimate, right: PortfolioEstimate) -> float:
    left_weights = _role_weights(left)
    right_weights = _role_weights(right)
    shared = sorted(set(left_weights) & set(right_weights))
    if not shared:
        return 0.0
    dot = sum(left_weights[market] * right_weights[market] for market in shared)
    left_norm = float(np.sqrt(sum(value**2 for value in left_weights.values())))
    right_norm = float(np.sqrt(sum(value**2 for value in right_weights.values())))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return float(np.clip(dot / (left_norm * right_norm), -1.0, 1.0))


def _role_overlap_sets(
    left: PortfolioEstimate,
    right: PortfolioEstimate,
) -> tuple[list[str], list[str]]:
    left_t = set(left.treatment_markets)
    left_c = set(left.control_markets)
    right_t = set(right.treatment_markets)
    right_c = set(right.control_markets)
    same_role = sorted((left_t & right_t) | (left_c & right_c))
    opposite_role = sorted((left_t & right_c) | (left_c & right_t))
    return same_role, opposite_role


def _pair_key(left_key: str, right_key: str) -> str:
    return f"{left_key}|{right_key}"


def _covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diag(covariance)
    scale = np.sqrt(np.outer(diagonal, diagonal))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.divide(covariance, scale, out=np.zeros_like(covariance), where=scale > 0)
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)


def _project_correlation_if_needed(correlation: np.ndarray) -> np.ndarray:
    symmetric = (correlation + correlation.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(np.min(eigenvalues)) >= -1e-10:
        return correlation
    clipped = np.clip(eigenvalues, 1e-12, None)
    projected = (eigenvectors * clipped) @ eigenvectors.T
    diagonal = np.sqrt(np.clip(np.diag(projected), 1e-12, None))
    projected = projected / np.outer(diagonal, diagonal)
    np.fill_diagonal(projected, 1.0)
    return np.clip((projected + projected.T) / 2.0, -1.0, 1.0)
