"""Helpers for running and summarizing multiple estimators."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from fieldtrial.estimators.advanced import SyntheticDIDEstimator
from fieldtrial.estimators.ascm import AugmentedSyntheticControlEstimator
from fieldtrial.estimators.base import CompletedDesign, Estimator, EstimatorResult, _jsonable
from fieldtrial.estimators.bayesian import BayesianTimeSeriesEstimator
from fieldtrial.estimators.bootstrap import BlockBootstrapEstimator
from fieldtrial.estimators.cuped import CUPEDAdjustedEstimator
from fieldtrial.estimators.did import DifferenceInDifferencesEstimator
from fieldtrial.estimators.forecast import ForecastCounterfactualEstimator
from fieldtrial.estimators.iroas import PairedIROASEstimator
from fieldtrial.estimators.matrix_completion import (
    GeneralizedSyntheticControlEstimator,
    MatrixCompletionEstimator,
)
from fieldtrial.estimators.ratio_delta import RatioDeltaEstimator
from fieldtrial.estimators.synthetic_control import SyntheticControlEstimator
from fieldtrial.estimators.tbr import TimeBasedRegressionEstimator
from fieldtrial.exceptions import OptionalDependencyError
from fieldtrial.inference.orchestration import (
    _validate_inference_spec,
    apply_configured_multiplicity,
    enrich_result_with_configured_methodology,
)
from fieldtrial.methods import family_consensus

ESTIMATOR_FACTORIES = {
    "did": DifferenceInDifferencesEstimator,
    "difference_in_differences": DifferenceInDifferencesEstimator,
    "ratio_delta": RatioDeltaEstimator,
    "block_bootstrap": BlockBootstrapEstimator,
    "forecast": ForecastCounterfactualEstimator,
    "forecast_counterfactual": ForecastCounterfactualEstimator,
    "cuped": CUPEDAdjustedEstimator,
    "ancova": CUPEDAdjustedEstimator,
    "residualized_did": CUPEDAdjustedEstimator,
    "synthetic_control": SyntheticControlEstimator,
    "synthetic_did": SyntheticDIDEstimator,
    "bayesian_time_series": BayesianTimeSeriesEstimator,
    "bayesian": BayesianTimeSeriesEstimator,
    "ascm": AugmentedSyntheticControlEstimator,
    "augmented_synthetic_control": AugmentedSyntheticControlEstimator,
    "matrix_completion": MatrixCompletionEstimator,
    "generalized_synthetic_control": GeneralizedSyntheticControlEstimator,
    "gsc": GeneralizedSyntheticControlEstimator,
    "tbr": TimeBasedRegressionEstimator,
    "time_based_regression": TimeBasedRegressionEstimator,
    "paired_iroas": PairedIROASEstimator,
    "iroas": PairedIROASEstimator,
}


@dataclass
class AnalysisResult:
    """Container for a completed experiment's multi-estimator analysis."""

    design: CompletedDesign
    metric: str
    results: list[EstimatorResult]
    errors: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def consensus(self) -> dict[str, Any]:
        return family_consensus(self.results)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "design": self.design.to_dict(),
                "metric": self.metric,
                "results": [result.to_dict() for result in self.results],
                "errors": self.errors,
                "metadata": self.metadata,
                "consensus": self.consensus(),
            }
        )


def default_estimators(*, include_bayesian: bool = False) -> list[Estimator]:
    estimators: list[Estimator] = [
        DifferenceInDifferencesEstimator(),
        RatioDeltaEstimator(),
        CUPEDAdjustedEstimator(),
        ForecastCounterfactualEstimator(),
        SyntheticDIDEstimator(),
        SyntheticControlEstimator(),
        BlockBootstrapEstimator(n_bootstrap=200, seed=0),
    ]
    if include_bayesian:
        estimators.append(BayesianTimeSeriesEstimator(draws=1000, seed=0))
    return estimators


def valid_estimator_names() -> list[str]:
    return sorted(ESTIMATOR_FACTORIES)


def instantiate_estimator(
    name: str,
    *,
    backend: str | None = None,
    params: dict[str, Any] | None = None,
) -> Estimator:
    try:
        factory = ESTIMATOR_FACTORIES[name]
    except KeyError as exc:
        valid = ", ".join(valid_estimator_names())
        raise ValueError(f"Unknown estimator {name!r}. Valid estimators: {valid}") from exc
    kwargs = dict(params or {})
    if backend is None:
        try:
            return factory(**kwargs)
        except TypeError as exc:
            raise ValueError(f"Invalid parameters for estimator {name!r}: {kwargs}") from exc
    signature = inspect.signature(factory)
    if "backend" not in signature.parameters:
        raise ValueError(f"Estimator {name!r} does not support backend overrides")
    if "backend" in kwargs and kwargs["backend"] != backend:
        raise ValueError(
            f"Estimator {name!r} received conflicting backend values: "
            f"{kwargs['backend']!r} and {backend!r}"
        )
    kwargs["backend"] = backend
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise ValueError(f"Invalid parameters for estimator {name!r}: {kwargs}") from exc


def _estimator_params_for(
    estimator_name: str,
    estimator_params: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if estimator_name in estimator_params:
        return dict(estimator_params[estimator_name])
    factory = ESTIMATOR_FACTORIES.get(estimator_name)
    canonical_name = getattr(factory, "name", None)
    if canonical_name and canonical_name in estimator_params:
        return dict(estimator_params[canonical_name])
    return {}


def _metric_names_for_analysis(
    catalog: Any,
    spec: Any,
    *,
    metrics: Iterable[str] | str | None,
    run_all: bool,
) -> list[str]:
    if metrics is not None and run_all:
        raise ValueError("Pass either metrics=... or run_all=True, not both")
    if run_all:
        names = list(catalog.names)
    elif metrics is None:
        names = list(getattr(spec, "primary_metrics", []))
    elif isinstance(metrics, str):
        names = [metrics]
    else:
        names = [str(metric) for metric in metrics]
    names = list(dict.fromkeys(str(name) for name in names))
    if not names:
        raise ValueError("No metrics selected for completed experiment analysis")
    unknown = sorted(name for name in names if name not in catalog)
    if unknown:
        raise ValueError(f"Unknown analysis metric(s): {unknown}")
    return names


class EstimatorEnsemble:
    """Run a list of estimators and collect standard FieldTrial results."""

    def __init__(
        self,
        estimators: Iterable[Estimator] | None = None,
        *,
        continue_on_error: bool = True,
    ) -> None:
        self.estimators = list(estimators) if estimators is not None else default_estimators()
        self.continue_on_error = continue_on_error

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> AnalysisResult:
        results: list[EstimatorResult] = []
        errors: list[dict[str, str]] = []
        metric_name = getattr(
            metric, "name", metric if isinstance(metric, str) else metric.__class__.__name__
        )
        for estimator in self.estimators:
            try:
                results.append(estimator.fit(panel, design, metric))
            except (OptionalDependencyError, NotImplementedError) as exc:
                errors.append(self._error_payload(estimator, exc))
                if not self.continue_on_error:
                    raise
            except Exception as exc:
                errors.append(self._error_payload(estimator, exc))
                if not self.continue_on_error:
                    raise
        return AnalysisResult(
            design=design,
            metric=str(metric_name),
            results=results,
            errors=errors,
        )

    @staticmethod
    def _error_payload(estimator: Estimator, exc: Exception) -> dict[str, str]:
        return {
            "estimator_name": getattr(estimator, "name", estimator.__class__.__name__),
            "error_type": exc.__class__.__name__,
            "message": str(exc),
        }


def analyze_completed_experiment(
    panel: Any,
    spec: Any,
    *,
    estimators: list[str] | None = None,
    metrics: Iterable[str] | str | None = None,
    run_all: bool = False,
    return_errors: bool = False,
    geo_col: str | None = None,
    time_col: str | None = None,
) -> list[EstimatorResult] | tuple[list[EstimatorResult], list[dict[str, str]]]:
    """Compatibility helper used by the CLI/integration workflow."""

    from fieldtrial.metrics.catalog import MetricCatalog

    suite = getattr(spec, "estimator_suite", None)
    estimator_names = estimators or list(
        getattr(
            suite,
            "estimators",
            [
                "did",
                "ratio_delta",
                "synthetic_did",
                "block_bootstrap",
                "synthetic_control",
            ],
        )
    )
    backend_overrides = getattr(suite, "backend_overrides", {}) or {}
    estimator_params = getattr(suite, "estimator_params", {}) or {}
    catalog = MetricCatalog.from_configs(spec.metrics)
    resolved_geo_col = geo_col or getattr(panel, "geo_col", "geo_id")
    resolved_time_col = time_col or getattr(panel, "time_col", "date")
    design = CompletedDesign(
        experiment_id=spec.experiment_id,
        treatment_geos=spec.treatment_geos,
        control_geos=spec.control_geos,
        start_date=spec.start_date,
        end_date=spec.end_date,
        pre_period_start=getattr(spec, "pre_period_start", None),
        pre_period_end=getattr(spec, "pre_period_end", None),
        geo_col=resolved_geo_col,
        time_col=resolved_time_col,
        metadata={"test_framework": spec.test_framework.model_dump(mode="json")},
    )
    _validate_inference_spec(spec.inference)
    metric_names = _metric_names_for_analysis(
        catalog,
        spec,
        metrics=metrics,
        run_all=run_all,
    )
    results: list[EstimatorResult] = []
    errors: list[dict[str, str]] = []
    for metric_name in metric_names:
        metric = catalog.get(metric_name)
        for estimator_name in estimator_names:
            try:
                estimator = instantiate_estimator(
                    estimator_name,
                    backend=backend_overrides.get(estimator_name),
                    params=_estimator_params_for(estimator_name, estimator_params),
                )
                result = estimator.fit(panel, design, metric)
                results.append(
                    enrich_result_with_configured_methodology(
                        panel,
                        design,
                        metric,
                        estimator,
                        result,
                        spec,
                    )
                )
            except Exception as exc:
                if not return_errors:
                    raise
                errors.append(
                    {
                        "metric": str(metric_name),
                        "estimator_name": estimator_name,
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    }
                )
    results = apply_configured_multiplicity(results, spec)
    if return_errors:
        return results, errors
    return results
