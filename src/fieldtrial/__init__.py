"""FieldTrial: geo experiment portfolio planning and measurement.

The public API is loaded lazily so the command-line entry point can remain usable while
optional implementation modules or estimator backends are being installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AssignmentMatrix": ("fieldtrial.design.assignments", "AssignmentMatrix"),
    "AssignmentPolicy": ("fieldtrial.design.policies", "AssignmentPolicy"),
    "AugmentedSyntheticControlEstimator": (
        "fieldtrial.estimators.ascm",
        "AugmentedSyntheticControlEstimator",
    ),
    "CandidateDesign": ("fieldtrial.design.candidates", "CandidateDesign"),
    "CandidateGenerator": ("fieldtrial.design.candidates", "CandidateGenerator"),
    "CalibrationResult": ("fieldtrial.methods", "CalibrationResult"),
    "CompletedDesign": ("fieldtrial.estimators.base", "CompletedDesign"),
    "CompositeMetric": ("fieldtrial.metrics", "CompositeMetric"),
    "ContinuousMetric": ("fieldtrial.metrics", "ContinuousMetric"),
    "CountMetric": ("fieldtrial.metrics", "CountMetric"),
    "CUPEDAdjustedEstimator": ("fieldtrial.estimators.cuped", "CUPEDAdjustedEstimator"),
    "EvidenceStore": ("fieldtrial.portfolio", "EvidenceStore"),
    "EstimandSpec": ("fieldtrial.methods", "EstimandSpec"),
    "EstimatorResult": ("fieldtrial.estimators.base", "EstimatorResult"),
    "evaluate_portfolio_decision": ("fieldtrial.portfolio", "evaluate_portfolio_decision"),
    "recommend_roadmap_actions": ("fieldtrial.portfolio", "recommend_roadmap_actions"),
    "InferenceResult": ("fieldtrial.methods", "InferenceResult"),
    "MatrixCompletionEstimator": (
        "fieldtrial.estimators.matrix_completion",
        "MatrixCompletionEstimator",
    ),
    "MethodMetadata": ("fieldtrial.methods", "MethodMetadata"),
    "MethodRegistry": ("fieldtrial.methods", "MethodRegistry"),
    "ExperimentRegistry": ("fieldtrial.registry.store", "ExperimentRegistry"),
    "ExperimentSpec": ("fieldtrial.design.specs", "ExperimentSpec"),
    "GeoPanel": ("fieldtrial.data.panel", "GeoPanel"),
    "generate_synthetic_panel": ("fieldtrial.data.synthetic", "generate_synthetic_panel"),
    "generate_synthetic_us_panel": ("fieldtrial.data.synthetic", "generate_synthetic_us_panel"),
    "ForecastCounterfactualEstimator": (
        "fieldtrial.estimators.forecast",
        "ForecastCounterfactualEstimator",
    ),
    "GeneralizedSyntheticControlEstimator": (
        "fieldtrial.estimators.matrix_completion",
        "GeneralizedSyntheticControlEstimator",
    ),
    "MetricCatalog": ("fieldtrial.metrics", "MetricCatalog"),
    "PortfolioPlanner": ("fieldtrial.optimize.portfolio", "PortfolioPlanner"),
    "PortfolioCovariance": ("fieldtrial.portfolio", "PortfolioCovariance"),
    "PortfolioEstimate": ("fieldtrial.portfolio", "PortfolioEstimate"),
    "PortfolioObjectiveWeights": ("fieldtrial.portfolio", "PortfolioObjectiveWeights"),
    "PortfolioSolution": ("fieldtrial.optimize.portfolio", "PortfolioSolution"),
    "PairedIROASEstimator": ("fieldtrial.estimators.iroas", "PairedIROASEstimator"),
    "RatioMetric": ("fieldtrial.metrics", "RatioMetric"),
    "RoadmapSpec": ("fieldtrial.design.specs", "RoadmapSpec"),
    "TimeBasedRegressionEstimator": (
        "fieldtrial.estimators.tbr",
        "TimeBasedRegressionEstimator",
    ),
    "estimate_cross_test_covariance": (
        "fieldtrial.portfolio",
        "estimate_cross_test_covariance",
    ),
    "score_candidate_portfolio": ("fieldtrial.portfolio", "score_candidate_portfolio"),
    "randomization_test": ("fieldtrial.inference", "randomization_test"),
    "split_conformal_counterfactual_interval": (
        "fieldtrial.inference",
        "split_conformal_counterfactual_interval",
    ),
    "summarize_roadmap_monitoring": ("fieldtrial.portfolio", "summarize_roadmap_monitoring"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'fieldtrial' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"{name} is not available because {module_name!r} could not be imported. "
            "Install the full package or finish the corresponding implementation module."
        ) from exc
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
