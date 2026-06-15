"""Native inference helpers for completed geo experiments.

The functions exported here are deliberately lightweight and dependency-light:
they accept plain arrays, mappings, pandas objects, or future assignment-policy
objects, then return the shared :class:`fieldtrial.methods.InferenceResult`
contract used by reports and estimators.
"""

from __future__ import annotations

from fieldtrial.inference.conformal import (
    conformal_counterfactual_test_inversion,
    split_conformal_counterfactual_interval,
)
from fieldtrial.inference.intervals import (
    bca_interval,
    cumulative_residual_interval,
    empirical_quantile_interval,
    fieller_interval,
    normal_interval,
    normal_p_value,
    t_interval,
    t_p_value,
    welch_difference_in_means,
)
from fieldtrial.inference.multiplicity import adjust_p_values, max_t_stepdown
from fieldtrial.inference.randomization import (
    FixedTreatmentCountPolicy,
    difference_in_means_statistic,
    randomization_test,
)
from fieldtrial.inference.resampling import (
    bootstrap_inference,
    jackknife_inference,
    leave_one_out_sensitivity,
    market_bootstrap,
)
from fieldtrial.inference.sequential import (
    bounded_mean_confidence_sequence,
    e_value_sequence,
)

_ORCHESTRATION_EXPORTS = {
    "analysis_methodology_status",
    "apply_configured_multiplicity",
    "enrich_result_with_configured_methodology",
}

__all__ = [
    "FixedTreatmentCountPolicy",
    "adjust_p_values",
    "analysis_methodology_status",
    "apply_configured_multiplicity",
    "bounded_mean_confidence_sequence",
    "bootstrap_inference",
    "bca_interval",
    "conformal_counterfactual_test_inversion",
    "cumulative_residual_interval",
    "difference_in_means_statistic",
    "e_value_sequence",
    "enrich_result_with_configured_methodology",
    "empirical_quantile_interval",
    "fieller_interval",
    "jackknife_inference",
    "leave_one_out_sensitivity",
    "market_bootstrap",
    "max_t_stepdown",
    "normal_interval",
    "normal_p_value",
    "randomization_test",
    "split_conformal_counterfactual_interval",
    "t_interval",
    "t_p_value",
    "welch_difference_in_means",
]


def __getattr__(name: str):
    if name in _ORCHESTRATION_EXPORTS:
        from fieldtrial.inference import orchestration

        value = getattr(orchestration, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
