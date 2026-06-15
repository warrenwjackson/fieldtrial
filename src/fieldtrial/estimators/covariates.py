"""Shared covariate preparation and screening helpers for estimators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CovariateFeature:
    """A numeric estimator feature derived from a user-facing covariate column."""

    source_column: str
    feature_column: str
    center: float
    std: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_column": self.source_column,
            "feature_column": self.feature_column,
            "center": self.center,
            "std": self.std,
        }


def prepare_covariate_features(
    frame: pd.DataFrame,
    covariate_columns: list[str] | tuple[str, ...],
    *,
    prefix: str,
    center_mask: pd.Series | np.ndarray | None = None,
    preserve_centered_names: bool = False,
    min_std: float = 1e-12,
) -> tuple[pd.DataFrame, list[CovariateFeature], list[dict[str, Any]]]:
    """Add centered numeric covariate features and report unusable candidates."""

    out = frame.copy()
    features: list[CovariateFeature] = []
    dropped: list[dict[str, Any]] = []
    used_names: set[str] = set(out.columns)

    if center_mask is None:
        basis_mask = pd.Series(True, index=out.index)
    else:
        basis_mask = pd.Series(center_mask, index=out.index).astype(bool)

    for index, raw_column in enumerate(dict.fromkeys(str(column) for column in covariate_columns)):
        if raw_column not in out.columns:
            dropped.append(
                {
                    "source_column": raw_column,
                    "feature_column": None,
                    "status": "dropped",
                    "reason": "missing_column",
                }
            )
            continue
        values = pd.to_numeric(out[raw_column], errors="coerce")
        if values.isna().any():
            dropped.append(
                {
                    "source_column": raw_column,
                    "feature_column": None,
                    "status": "dropped",
                    "reason": "missing_or_non_numeric_values",
                    "n_missing": int(values.isna().sum()),
                }
            )
            continue
        basis_values = values.loc[basis_mask]
        if basis_values.empty:
            basis_values = values
        center = float(basis_values.mean())
        std = float(basis_values.std(ddof=0))
        if not np.isfinite(std) or std <= min_std:
            dropped.append(
                {
                    "source_column": raw_column,
                    "feature_column": None,
                    "status": "dropped",
                    "reason": "zero_variance",
                    "std": std if np.isfinite(std) else None,
                }
            )
            continue
        feature_column = _feature_name(
            raw_column,
            prefix=prefix,
            index=index,
            used_names=used_names,
            preserve_centered_names=preserve_centered_names,
        )
        out[feature_column] = values - center
        used_names.add(feature_column)
        features.append(
            CovariateFeature(
                source_column=raw_column,
                feature_column=feature_column,
                center=center,
                std=std,
            )
        )

    return out, features, dropped


def select_covariates(
    frame: pd.DataFrame,
    *,
    outcome_col: str,
    candidate_features: list[CovariateFeature],
    base_columns: list[str] | tuple[str, ...] = (),
    evaluation_mask: pd.Series | np.ndarray | None = None,
    min_relative_improvement: float = 0.01,
    max_features: int | None = None,
) -> dict[str, Any]:
    """Forward-select covariates only while they improve predictive fit.

    The selector is deterministic. It prefers cross-validated RMSE and falls back
    to in-sample BIC only when there are too few usable evaluation rows for
    cross-validation. Lower scores are better for both strategies.
    """

    if min_relative_improvement < 0:
        raise ValueError("min_relative_improvement must be non-negative")
    if max_features is not None and max_features < 0:
        raise ValueError("max_features must be non-negative")

    if evaluation_mask is None:
        evaluation = frame.copy()
    else:
        mask = pd.Series(evaluation_mask, index=frame.index).astype(bool)
        evaluation = frame.loc[mask].copy()

    candidate_columns = [feature.feature_column for feature in candidate_features]
    required = [outcome_col, *base_columns, *candidate_columns]
    evaluation = evaluation.dropna(subset=required)
    feature_records = {
        feature.feature_column: {
            **feature.to_dict(),
            "status": "candidate",
            "score": None,
            "relative_improvement": None,
            "reason": None,
        }
        for feature in candidate_features
    }
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "candidate_features": [feature.to_dict() for feature in candidate_features],
        "selected_features": [],
        "selected_source_columns": [],
        "rejected_features": [],
        "rejected_source_columns": [],
        "feature_decisions": [],
        "base_columns": list(base_columns),
        "evaluation_rows": int(len(evaluation)),
        "min_relative_improvement": float(min_relative_improvement),
        "max_features": max_features,
        "score_name": None,
        "baseline_score": None,
        "final_score": None,
    }
    if not candidate_features:
        diagnostics["strategy"] = "no_candidates"
        return diagnostics
    if evaluation.empty:
        diagnostics["strategy"] = "no_evaluation_rows"
        for record in feature_records.values():
            record.update({"status": "rejected", "reason": "no_evaluation_rows"})
        diagnostics["feature_decisions"] = list(feature_records.values())
        diagnostics["rejected_features"] = candidate_columns
        diagnostics["rejected_source_columns"] = [
            feature.source_column for feature in candidate_features
        ]
        return diagnostics

    y = evaluation[outcome_col].to_numpy(dtype=float)
    selected: list[CovariateFeature] = []
    remaining = list(candidate_features)
    limit = len(remaining) if max_features is None else min(max_features, len(remaining))
    base_matrix = _design_matrix(evaluation, base_columns)
    score_name = (
        "cross_validated_rmse"
        if _can_cross_validate(len(evaluation), base_matrix.shape[1])
        else "in_sample_bic"
    )
    current_score = _score(y, base_matrix, score_name)
    diagnostics["strategy"] = "forward_selection"
    diagnostics["score_name"] = score_name
    diagnostics["baseline_score"] = current_score
    if current_score is None:
        for record in feature_records.values():
            record.update({"status": "rejected", "reason": "insufficient_degrees_of_freedom"})
        diagnostics["feature_decisions"] = list(feature_records.values())
        diagnostics["rejected_features"] = candidate_columns
        diagnostics["rejected_source_columns"] = [
            feature.source_column for feature in candidate_features
        ]
        return diagnostics

    while remaining and len(selected) < limit:
        best_feature: CovariateFeature | None = None
        best_score: float | None = None
        best_relative_improvement: float | None = None
        for feature in remaining:
            columns = [
                *base_columns,
                *[item.feature_column for item in selected],
                feature.feature_column,
            ]
            matrix = _design_matrix(evaluation, columns)
            score = _score(y, matrix, score_name)
            if score is None:
                feature_records[feature.feature_column].update(
                    {"score": None, "reason": "insufficient_degrees_of_freedom"}
                )
                continue
            relative_improvement = _relative_improvement(current_score, score)
            feature_records[feature.feature_column].update(
                {
                    "score": float(score),
                    "relative_improvement": float(relative_improvement),
                    "reason": "evaluated",
                }
            )
            if best_score is None or score < best_score:
                best_feature = feature
                best_score = score
                best_relative_improvement = relative_improvement

        if (
            best_feature is None
            or best_score is None
            or best_relative_improvement is None
            or best_relative_improvement < min_relative_improvement
        ):
            break

        selected.append(best_feature)
        remaining = [feature for feature in remaining if feature != best_feature]
        current_score = best_score
        feature_records[best_feature.feature_column].update(
            {
                "status": "selected",
                "score": float(best_score),
                "relative_improvement": float(best_relative_improvement),
                "reason": "improved_score",
                "selection_order": len(selected),
            }
        )

    selected_columns = [feature.feature_column for feature in selected]
    selected_sources = [feature.source_column for feature in selected]
    for feature in remaining:
        record = feature_records[feature.feature_column]
        if record["reason"] == "evaluated":
            record["reason"] = "no_incremental_improvement"
        record["status"] = "rejected"

    diagnostics["selected_features"] = selected_columns
    diagnostics["selected_source_columns"] = selected_sources
    diagnostics["rejected_features"] = [
        feature.feature_column
        for feature in candidate_features
        if feature.feature_column not in selected_columns
    ]
    diagnostics["rejected_source_columns"] = [
        feature.source_column
        for feature in candidate_features
        if feature.feature_column not in selected_columns
    ]
    diagnostics["feature_decisions"] = list(feature_records.values())
    diagnostics["final_score"] = current_score
    return diagnostics


def _feature_name(
    column: str,
    *,
    prefix: str,
    index: int,
    used_names: set[str],
    preserve_centered_names: bool,
) -> str:
    if preserve_centered_names:
        candidate = f"{column}_centered"
        if candidate not in used_names:
            return candidate
    slug = re.sub(r"\W+", "_", column).strip("_").lower() or "covariate"
    candidate = f"{prefix}_{index}_{slug}"
    if candidate not in used_names:
        return candidate
    suffix = 1
    while f"{candidate}_{suffix}" in used_names:
        suffix += 1
    return f"{candidate}_{suffix}"


def _design_matrix(frame: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> np.ndarray:
    if columns:
        values = frame.loc[:, list(columns)].to_numpy(dtype=float)
        return np.column_stack([np.ones(len(frame)), values])
    return np.ones((len(frame), 1), dtype=float)


def _can_cross_validate(n_rows: int, n_parameters: int) -> bool:
    return n_rows >= 4 and n_rows >= n_parameters + 2


def _score(y: np.ndarray, x: np.ndarray, score_name: str) -> float | None:
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        return None
    if score_name == "cross_validated_rmse":
        return _cross_validated_rmse(y, x)
    return _bic(y, x)


def _cross_validated_rmse(y: np.ndarray, x: np.ndarray) -> float | None:
    n_rows, n_parameters = x.shape
    if not _can_cross_validate(n_rows, n_parameters):
        return None
    n_folds = n_rows if n_rows <= 25 else min(5, n_rows)
    predictions = np.empty(n_rows, dtype=float)
    predictions.fill(np.nan)
    for test_index in np.array_split(np.arange(n_rows), n_folds):
        if len(test_index) == 0:
            continue
        train_mask = np.ones(n_rows, dtype=bool)
        train_mask[test_index] = False
        if int(train_mask.sum()) <= n_parameters:
            return None
        coefficients = np.linalg.lstsq(x[train_mask], y[train_mask], rcond=None)[0]
        predictions[test_index] = x[test_index] @ coefficients
    if not np.all(np.isfinite(predictions)):
        return None
    return float(np.sqrt(np.mean((y - predictions) ** 2)))


def _bic(y: np.ndarray, x: np.ndarray) -> float | None:
    n_rows, n_parameters = x.shape
    if n_rows <= n_parameters:
        return None
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    rss = float(np.sum(residuals**2))
    if not np.isfinite(rss):
        return None
    rss = max(rss, 1e-24)
    return float(n_rows * np.log(rss / n_rows) + n_parameters * np.log(n_rows))


def _relative_improvement(previous_score: float, candidate_score: float) -> float:
    improvement = previous_score - candidate_score
    return float(improvement / max(abs(previous_score), 1e-12))
