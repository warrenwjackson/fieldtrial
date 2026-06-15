from fieldtrial.estimators.advanced import SyntheticDIDEstimator
from fieldtrial.estimators.ascm import AugmentedSyntheticControlEstimator
from fieldtrial.estimators.base import CompletedDesign, EstimatorResult
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
from fieldtrial.estimators.tbr import TBREstimator, TbrEstimator, TimeBasedRegressionEstimator

__all__ = [
    "AugmentedSyntheticControlEstimator",
    "BayesianTimeSeriesEstimator",
    "BlockBootstrapEstimator",
    "CompletedDesign",
    "CUPEDAdjustedEstimator",
    "DifferenceInDifferencesEstimator",
    "EstimatorResult",
    "ForecastCounterfactualEstimator",
    "GeneralizedSyntheticControlEstimator",
    "MatrixCompletionEstimator",
    "PairedIROASEstimator",
    "RatioDeltaEstimator",
    "SyntheticDIDEstimator",
    "SyntheticControlEstimator",
    "TBREstimator",
    "TbrEstimator",
    "TimeBasedRegressionEstimator",
    "analyze_completed_experiment",
]


def __getattr__(name: str):
    if name == "analyze_completed_experiment":
        from fieldtrial.estimators.ensemble import analyze_completed_experiment

        return analyze_completed_experiment
    raise AttributeError(name)
