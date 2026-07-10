"""Execution layer for configured completed-test methodology."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from fieldtrial.calibration.injection import injected_lift_recovery_curve
from fieldtrial.calibration.placebo import (
    PLACEBO_IN_SPACE,
    PLACEBO_IN_TIME,
    not_applicable_placebo_result,
    placebo_applicability,
    placebo_backtest,
    placebo_in_space_backtest,
)
from fieldtrial.design.policies import AssignmentPolicy
from fieldtrial.estimators.base import (
    OUTCOME_COL,
    PERIOD_COL,
    CompletedDesign,
    Estimator,
    EstimatorResult,
    prepare_estimator_frame,
)
from fieldtrial.inference.conformal import (
    conformal_counterfactual_test_inversion,
    split_conformal_counterfactual_interval,
)
from fieldtrial.inference.multiplicity import adjust_p_values, max_t_stepdown
from fieldtrial.inference.randomization import randomization_test
from fieldtrial.inference.resampling import bootstrap_inference, jackknife_inference
from fieldtrial.inference.sequential import bounded_mean_confidence_sequence
from fieldtrial.methods import CalibrationResult, EstimandSpec, InferenceResult

ESTIMATOR_DEFAULT = "estimator_default"
SUPPORTED_INFERENCE_METHODS = {
    ESTIMATOR_DEFAULT,
    "randomization_inference",
    "market_bootstrap",
    "block_bootstrap",
    "bootstrap",
    "jackknife",
    "conformal_inference",
    "split_conformal",
    "few_cluster_robust",
}
SUPPORTED_MULTIPLICITY = {
    "none",
    "bonferroni",
    "holm",
    "benjamini_hochberg",
    "westfall_young",
}


def enrich_result_with_configured_methodology(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    estimator: Estimator,
    result: EstimatorResult,
    spec: Any,
) -> EstimatorResult:
    """Attach requested inference, calibration, and monitoring outputs."""

    _validate_inference_spec(spec.inference)
    inference_results = list(result.inference_results)
    native_inference_count = len(inference_results)
    for method in spec.inference.methods:
        canonical = _canonical_inference_method(method)
        if canonical == ESTIMATOR_DEFAULT:
            continue
        inference_results.append(
            _run_inference_method(panel, design, metric, result, spec, method=canonical)
        )

    monitoring_result = _run_monitoring_if_requested(panel, design, metric, spec)
    if monitoring_result is not None:
        inference_results.append(monitoring_result)

    calibration_results = [
        *result.calibration_results,
        *_run_calibration_if_requested(panel, design, metric, estimator, spec),
    ]
    enriched = replace(
        result,
        inference_results=inference_results,
        calibration_results=calibration_results,
    )
    return _promote_primary_inference(
        enriched,
        preferred_method=_canonical_inference_method(spec.inference.primary_method),
        native_inference_count=native_inference_count,
    )


def _promote_primary_inference(
    result: EstimatorResult,
    *,
    preferred_method: str | None = None,
    native_inference_count: int | None = None,
) -> EstimatorResult:
    selected_index: int | None = None
    selected_rank = -1
    for index, inference in enumerate(result.inference_results):
        if preferred_method == ESTIMATOR_DEFAULT and native_inference_count is not None:
            if index >= native_inference_count:
                continue
        elif preferred_method not in {None, ESTIMATOR_DEFAULT}:
            configured_method = str(
                inference.diagnostics.get("configured_inference_method") or inference.method
            )
            if configured_method != preferred_method:
                continue
        if not _inference_is_primary_compatible(result, inference):
            continue
        rank = _primary_inference_rank(inference)
        if rank > selected_rank:
            selected_index = index
            selected_rank = rank
    if selected_index is None or selected_rank <= 0:
        if preferred_method not in {None, ESTIMATOR_DEFAULT}:
            return replace(
                result,
                diagnostics={
                    **result.diagnostics,
                    "requested_primary_inference": preferred_method,
                    "primary_inference_status": "not_promoted_estimand_mismatch_or_unavailable",
                },
                warnings=[
                    *result.warnings,
                    (
                        f"Configured primary inference {preferred_method!r} did not return an "
                        "interval/statistic compatible with this estimator's exact estimand; "
                        "the estimator-native uncertainty remains in the top-level fields."
                    ),
                ],
            )
        return result
    selected = result.inference_results[selected_index]
    inference_results = []
    for index, inference in enumerate(result.inference_results):
        inference_results.append(
            replace(
                inference,
                diagnostics={
                    **inference.diagnostics,
                    "selected_as_primary": index == selected_index,
                },
            )
        )
    diagnostics = {
        **result.diagnostics,
        "primary_inference_method": selected.method,
        "primary_interval_type": selected.interval_type,
        "primary_inference_family": selected.method_family,
    }
    return replace(
        result,
        interval=selected.interval,
        p_value=selected.p_value,
        primary_adjusted_p_value=selected.adjusted_p_value,
        decision_p_value=(
            selected.adjusted_p_value if selected.adjusted_p_value is not None else selected.p_value
        ),
        standard_error=selected.standard_error,
        relative_interval=None,
        diagnostics=diagnostics,
        inference_results=inference_results,
    )


def _inference_is_primary_compatible(
    result: EstimatorResult,
    inference: InferenceResult,
) -> bool:
    """Whether an inference payload targets the result's exact estimand contract."""

    if inference.primary_eligible is False:
        return False
    if inference.estimand_spec is not None:
        inference_spec = EstimandSpec.coerce(inference.estimand_spec, metric=result.metric)
        if not result.estimand_spec.compatible_with(inference_spec):
            return False
        if inference.point_estimate is None:
            return False
        return bool(
            np.isclose(
                float(inference.point_estimate),
                float(result.estimate),
                rtol=1e-7,
                atol=max(1e-10, abs(float(result.estimate)) * 1e-9),
            )
        )

    # Backward-compatible native inference: an unannotated payload is eligible
    # only when it exactly reproduces fields already returned by the estimator.
    interval_matches = inference.interval == result.interval
    p_matches = inference.p_value == result.p_value
    se_matches = inference.standard_error == result.standard_error
    point_matches = inference.point_estimate is None or np.isclose(
        float(inference.point_estimate),
        float(result.estimate),
        rtol=1e-7,
        atol=max(1e-10, abs(float(result.estimate)) * 1e-9),
    )
    return bool(point_matches and interval_matches and (p_matches or se_matches))


def _primary_inference_rank(inference: InferenceResult) -> int:
    if inference.method_family == "multiplicity":
        return 0
    interval_type = str(inference.interval_type or "")
    method = str(inference.method)
    if interval_type == "randomization_test_inversion":
        return 100
    if method == "randomization_inference":
        return 95
    if interval_type == "wild_cluster_bootstrap_t":
        return 90
    if interval_type == "moving_block_conformal_inversion":
        return 85
    if interval_type == "split_conformal_cumulative_effect":
        return 75
    if interval_type == "bca_bootstrap":
        return 70
    if interval_type == "jackknife_t":
        return 65
    if interval_type == "bootstrap_percentile":
        return 55
    if inference.interval is not None or inference.p_value is not None:
        return 10
    return 0


def apply_configured_multiplicity(
    results: Sequence[EstimatorResult],
    spec: Any,
) -> list[EstimatorResult]:
    """Append configured multiplicity corrections across estimator/metric hypotheses."""

    method = str(spec.inference.multiplicity)
    if method == "none":
        return list(results)
    if method == "westfall_young":
        return _apply_westfall_young_multiplicity(results, spec)
    if method not in SUPPORTED_MULTIPLICITY:
        raise ValueError(
            "Unsupported inference.multiplicity "
            f"{method!r}. Supported values: {', '.join(sorted(SUPPORTED_MULTIPLICITY))}."
        )

    p_values: dict[str, float] = {}
    result_indexes: dict[str, int] = {}
    primary_indexes = [
        index
        for index, result in enumerate(results)
        if bool(result.diagnostics.get("is_primary_estimator"))
    ]
    included_indexes = primary_indexes or list(range(len(results)))
    for index in included_indexes:
        result = results[index]
        p_value = _primary_p_value(result)
        if p_value is None:
            continue
        hypothesis_id = f"{result.metric}:{result.estimator_name}"
        p_values[hypothesis_id] = p_value
        result_indexes[hypothesis_id] = index
    if not p_values:
        raise ValueError(
            "Multiplicity correction was requested, but no estimator produced a finite p-value."
        )

    adjusted = adjust_p_values(
        p_values,
        method=method,
        alpha=1.0 - float(spec.inference.confidence),
    )
    output = list(results)
    for inference in adjusted:
        hypothesis_id = str(inference.diagnostics["hypothesis_id"])
        index = result_indexes[hypothesis_id]
        existing = list(output[index].inference_results)
        output[index] = _with_multiplicity_p_value(
            replace(output[index], inference_results=[*existing, inference]),
            inference,
        )
    return output


def _apply_westfall_young_multiplicity(
    results: Sequence[EstimatorResult],
    spec: Any,
) -> list[EstimatorResult]:
    selected_indexes = [
        index
        for index, result in enumerate(results)
        if bool(result.diagnostics.get("is_primary_estimator"))
    ]
    if not selected_indexes:
        selected_indexes = list(range(len(results)))
    selected_results = [results[index] for index in selected_indexes]
    payloads = [_max_t_payload(result) for result in selected_results]
    if any(payload is None for payload in payloads):
        raise ValueError(
            "inference.multiplicity='westfall_young' requires every result to carry aligned "
            "stored null draws. Add randomization_inference or market_bootstrap to "
            "inference.methods and keep draw storage enabled."
        )
    typed_payloads = [payload for payload in payloads if payload is not None]
    sources = {payload["source"] for payload in typed_payloads}
    lengths = {len(payload["null_statistics"]) for payload in typed_payloads}
    if len(sources) != 1 or len(lengths) != 1:
        raise ValueError(
            "inference.multiplicity='westfall_young' requires all hypotheses to use the "
            "same aligned null-draw source and draw count."
        )
    hypothesis_ids = [f"{result.metric}:{result.estimator_name}" for result in selected_results]
    observed_statistics = {
        hypothesis_id: float(payload["observed_statistic"])
        for hypothesis_id, payload in zip(hypothesis_ids, typed_payloads, strict=True)
    }
    joint_null = np.column_stack([payload["null_statistics"] for payload in typed_payloads])
    adjusted = max_t_stepdown(
        observed_statistics,
        joint_null,
        alpha=1.0 - float(spec.inference.confidence),
        two_sided=True,
    )
    output = list(results)
    for index, inference in enumerate(adjusted):
        output_index = selected_indexes[index]
        existing = list(output[output_index].inference_results)
        promoted_inference = replace(
            inference,
            diagnostics={
                **inference.diagnostics,
                "hypothesis_id": hypothesis_ids[index],
                "null_draw_source": typed_payloads[index]["source"],
                "studentized": True,
            },
        )
        output[output_index] = _with_multiplicity_p_value(
            replace(
                output[output_index],
                inference_results=[*existing, promoted_inference],
            ),
            promoted_inference,
        )
    return output


def _with_multiplicity_p_value(
    result: EstimatorResult,
    inference: InferenceResult,
) -> EstimatorResult:
    adjusted = _finite(inference.adjusted_p_value)
    raw = _finite(inference.p_value)
    if adjusted is None:
        return result
    return replace(
        result,
        primary_adjusted_p_value=adjusted,
        decision_p_value=adjusted,
        p_value=result.p_value if result.p_value is not None else raw,
        diagnostics={
            **result.diagnostics,
            "decision_p_value_source": inference.method,
            "multiplicity_method": inference.method,
        },
    )


def _max_t_payload(result: EstimatorResult) -> dict[str, Any] | None:
    for inference in result.inference_results:
        if inference.method == "randomization_inference":
            draws = inference.artifacts.get("null_statistics")
            if draws is None:
                continue
            observed = _finite_number((inference.null_distribution or {}).get("observed_statistic"))
            null_value = (
                _finite_number((inference.null_distribution or {}).get("null_value")) or 0.0
            )
            payload = _studentized_null_payload(
                draws,
                observed=observed,
                center=null_value,
                source="randomization_inference",
            )
            if payload is not None:
                return payload
        if inference.method in {"market_bootstrap", "bootstrap"}:
            draws = inference.artifacts.get("bootstrap_statistics")
            if draws is None:
                continue
            observed = _finite_number((inference.null_distribution or {}).get("observed_statistic"))
            # Bootstrap-t null approximation: the draws are centered at the
            # observed estimate to form the null spread, but the observed
            # statistic must be studentized against the hypothesis null (0) -
            # studentizing it against itself makes the observed t exactly 0 and
            # every Westfall-Young adjusted p-value ~1.
            payload = _studentized_null_payload(
                draws,
                observed=observed,
                draw_center=observed,
                null_center=0.0,
                source=inference.method,
            )
            if payload is not None:
                return payload
    return None


def _studentized_null_payload(
    draws: Any,
    *,
    observed: float | None,
    center: float | None = None,
    draw_center: float | None = None,
    null_center: float | None = None,
    source: str,
) -> dict[str, Any] | None:
    if center is not None:
        draw_center = center if draw_center is None else draw_center
        null_center = center if null_center is None else null_center
    if observed is None or draw_center is None or null_center is None:
        return None
    array = np.asarray(draws, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        return None
    scale = float(np.std(array - draw_center, ddof=1))
    if scale <= 0 or not np.isfinite(scale):
        return None
    return {
        "source": source,
        "observed_statistic": float((observed - null_center) / scale),
        "null_statistics": (array - draw_center) / scale,
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def analysis_methodology_status(results: Sequence[EstimatorResult], spec: Any) -> dict[str, Any]:
    """Summarize configured methodology from artifacts that actually exist."""

    inference_methods = _unique(
        inference.method for result in results for inference in result.inference_results
    )
    calibration_results = [
        calibration for result in results for calibration in result.calibration_results
    ]
    calibration_methods = _unique(
        calibration.method
        for calibration in calibration_results
        if calibration.status != "not_applicable"
    )
    monitoring_methods = _unique(
        inference.method
        for result in results
        for inference in result.inference_results
        if inference.method_family in {"sequential", "monitoring"}
    )
    warnings = _unique(
        warning
        for result in results
        for warning in [
            *result.warnings,
            *[item for inference in result.inference_results for item in inference.warnings],
            *[item for calibration in result.calibration_results for item in calibration.warnings],
        ]
    )
    calibration_requested = _calibration_requested(spec)
    monitoring_requested = spec.monitoring.mode != "fixed_horizon"
    calibration_status = _calibration_status_payload(
        calibration_results,
        spec,
        calibration_requested=calibration_requested,
        run_methods=calibration_methods,
    )
    return {
        "assignment_policy": (
            None
            if spec.assignment_policy is None
            else spec.assignment_policy.model_dump(mode="json")
        ),
        "inference": {
            "configured": spec.inference.model_dump(mode="json"),
            "status": "run",
            "run_methods": inference_methods,
            "not_run_methods": [],
        },
        "calibration": calibration_status,
        "monitoring": {
            "configured": spec.monitoring.model_dump(mode="json"),
            "status": "run" if monitoring_requested else "not_requested",
            "run_methods": monitoring_methods,
            "not_run_methods": [],
        },
        "warnings": warnings,
    }


def _calibration_status_payload(
    calibration_results: Sequence[CalibrationResult],
    spec: Any,
    *,
    calibration_requested: bool,
    run_methods: list[str],
) -> dict[str, Any]:
    failures = [
        _calibration_status_record(calibration)
        for calibration in calibration_results
        if calibration.status == "fail"
    ]
    exclusions = [
        _calibration_status_record(calibration)
        for calibration in calibration_results
        if calibration.status == "not_applicable"
    ]
    cautionary = [
        _calibration_status_record(calibration)
        for calibration in calibration_results
        if calibration.status in {"warning", "not_evaluable"}
    ]
    if not calibration_requested:
        status = "not_requested"
    elif failures:
        status = "fail"
    elif cautionary:
        status = "warning"
    elif calibration_results and len(exclusions) == len(calibration_results):
        status = "not_applicable"
    elif calibration_results:
        status = "run"
    else:
        status = "not_evaluable"

    return {
        "configured": spec.calibration.model_dump(mode="json"),
        "status": status,
        "run_methods": run_methods,
        "not_run_methods": _unique(
            f"{record['estimator_name']}:{record['method']}" for record in exclusions
        ),
        "failures": failures,
        "warnings": cautionary,
        "exclusions": exclusions,
        "summary": {
            "result_count": len(calibration_results),
            "failure_count": len(failures),
            "warning_count": len(cautionary),
            "exclusion_count": len(exclusions),
        },
    }


def _calibration_status_record(calibration: CalibrationResult) -> dict[str, Any]:
    return {
        "metric": calibration.metric,
        "estimator_name": calibration.estimator_name,
        "method": calibration.method,
        "status": calibration.status,
        "status_reason": calibration.status_reason,
        "placebo_false_positive_rate": calibration.placebo_false_positive_rate,
        "coverage": calibration.coverage,
        "bias": calibration.bias,
        "rmse": calibration.rmse,
        "warning_rate": calibration.warning_rate,
    }


def _validate_inference_spec(inference_spec: Any) -> None:
    unknown = sorted(
        {
            str(method)
            for method in inference_spec.methods
            if _canonical_inference_method(str(method)) not in SUPPORTED_INFERENCE_METHODS
        }
    )
    if unknown:
        raise ValueError(
            "Unsupported inference method(s): "
            f"{', '.join(unknown)}. Supported methods: "
            f"{', '.join(sorted(SUPPORTED_INFERENCE_METHODS))}."
        )


def _canonical_inference_method(method: str) -> str:
    key = method.lower().replace("-", "_")
    aliases = {
        "randomization": "randomization_inference",
        "permutation": "randomization_inference",
        "permutation_test": "randomization_inference",
        "block_bootstrap": "market_bootstrap",
        "bootstrap": "market_bootstrap",
        "split_conformal_counterfactual": "conformal_inference",
        "split_conformal": "conformal_inference",
        "conformal": "conformal_inference",
        "few_cluster": "few_cluster_robust",
        "small_sample": "few_cluster_robust",
    }
    return aliases.get(key, key)


def _run_inference_method(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    result: EstimatorResult,
    spec: Any,
    *,
    method: str,
) -> InferenceResult:
    if method == "randomization_inference":
        inference = _run_randomization_inference(panel, design, metric, result, spec)
    elif method == "market_bootstrap":
        inference = _run_market_bootstrap(panel, design, metric, result, spec)
    elif method == "jackknife":
        inference = _run_jackknife(panel, design, metric, result, spec)
    elif method == "conformal_inference":
        inference = _run_conformal(result, spec)
    elif method == "few_cluster_robust":
        inference = _run_few_cluster_wild_bootstrap(panel, design, metric, result, spec)
    else:
        raise ValueError(f"Unsupported inference method {method!r}")
    return replace(
        inference,
        diagnostics={**inference.diagnostics, "configured_inference_method": method},
    )


def _run_randomization_inference(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    result: EstimatorResult | Any,
    spec: Any | None = None,
) -> InferenceResult:
    if spec is None:
        spec = result
        result = None
    result_metric = getattr(result, "metric", getattr(metric, "name", str(metric)))
    source_estimator = getattr(result, "estimator_name", "configured_inference")
    effects = _market_effect_frame(panel, design, metric)
    policy = _assignment_policy_from_completed_spec(spec, design)
    n_feasible = policy.n_feasible_assignments if policy is not None else None
    n_permutations = (
        spec.inference.randomization_samples
        if n_feasible is not None and n_feasible > spec.assignment_policy.max_enumerated_assignments
        else None
    )
    inference = randomization_test(
        dict(zip(effects["geo_id"], effects["effect"], strict=True)),
        treatment_units=design.treatment_geos,
        control_units=design.control_geos,
        policy=policy,
        alternative=_alternative_from_framework(spec),
        null_value=_null_value_from_framework(
            spec,
            metric,
            baseline=_effect_frame_baseline(effects),
        ),
        n_permutations=n_permutations,
        seed=_assignment_seed(spec),
        max_exact_assignments=_max_exact_assignments(spec),
        confidence=float(spec.inference.confidence),
    )
    return replace(
        inference,
        estimand_spec=_market_effect_estimand(metric, result_metric),
        point_estimate=_effect_frame_statistic(effects),
        primary_eligible=True,
        diagnostics={
            **inference.diagnostics,
            "source_estimator": source_estimator,
            "source_metric": result_metric,
        },
    )


def _run_market_bootstrap(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    result: EstimatorResult | Any,
    spec: Any | None = None,
) -> InferenceResult:
    if spec is None:
        spec = result
        result = None
    result_metric = getattr(result, "metric", getattr(metric, "name", str(metric)))
    source_estimator = getattr(result, "estimator_name", "configured_inference")
    effects = _market_effect_frame(panel, design, metric)
    n_treatment = int(effects.loc[effects["treated"].astype(bool), "geo_id"].nunique())
    n_control = int(effects.loc[~effects["treated"].astype(bool), "geo_id"].nunique())
    if min(n_treatment, n_control) < 2:
        return InferenceResult(
            method="market_bootstrap",
            method_family="bootstrap",
            confidence=float(spec.inference.confidence),
            estimand_spec=_market_effect_estimand(metric, result_metric),
            primary_eligible=False,
            assumptions=[
                "Resampling units are exchangeable within each treatment arm.",
                "At least two markets per arm are needed for arm-level bootstrap variation.",
            ],
            diagnostics={
                "n_treatment_markets": n_treatment,
                "n_control_markets": n_control,
                "status": "not_evaluable_fewer_than_two_markets_per_arm",
            },
            warnings=[
                "Market bootstrap was not promoted because one arm has fewer than two "
                "markets; use assignment-aware randomization inference for one-treated-geo "
                "designs."
            ],
        )
    inference = bootstrap_inference(
        effects,
        statistic=_effect_frame_statistic,
        unit_col="geo_id",
        strata_col="treated",
        n_resamples=int(spec.inference.bootstrap_samples),
        seed=_assignment_seed(spec),
        confidence=float(spec.inference.confidence),
        alternative=_alternative_from_framework(spec),
        null_value=_null_value_from_framework(
            spec,
            metric,
            baseline=_effect_frame_baseline(effects),
        ),
        method="market_bootstrap",
    )
    return replace(
        inference,
        estimand_spec=_market_effect_estimand(metric, result_metric),
        point_estimate=_effect_frame_statistic(effects),
        primary_eligible=True,
        diagnostics={
            **inference.diagnostics,
            "source_estimator": source_estimator,
            "source_metric": result_metric,
        },
    )


def _run_jackknife(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    result: EstimatorResult,
    spec: Any,
) -> InferenceResult:
    effects = _market_effect_frame(panel, design, metric)
    inference = jackknife_inference(
        effects,
        statistic=_effect_frame_statistic,
        unit_col="geo_id",
        confidence=float(spec.inference.confidence),
        alternative=_alternative_from_framework(spec),
        null_value=_null_value_from_framework(
            spec,
            metric,
            baseline=_effect_frame_baseline(effects),
        ),
    )
    return replace(
        inference,
        estimand_spec=_market_effect_estimand(metric, result.metric),
        point_estimate=_effect_frame_statistic(effects),
        primary_eligible=True,
        diagnostics={
            **inference.diagnostics,
            "source_estimator": result.estimator_name,
            "source_metric": result.metric,
        },
    )


def _run_conformal(result: EstimatorResult, spec: Any) -> InferenceResult:
    records = result.artifacts.get("counterfactual")
    if not isinstance(records, list) or not records:
        raise ValueError(
            "conformal_inference requires estimator counterfactual artifacts with pre and "
            "post observed/counterfactual paths."
        )
    pre_observed, pre_counterfactual, post_observed, post_counterfactual = _counterfactual_series(
        records
    )
    post_gaps = np.asarray(post_observed, dtype=float) - np.asarray(
        post_counterfactual,
        dtype=float,
    )
    pre_residuals = np.asarray(pre_observed, dtype=float) - np.asarray(
        pre_counterfactual,
        dtype=float,
    )
    fallback_warning: str | None = None
    null_value = _null_value_from_framework(
        spec,
        result.metric,
        baseline=_finite_number(np.sum(np.asarray(post_counterfactual, dtype=float))),
    )
    try:
        inference = conformal_counterfactual_test_inversion(
            post_gaps,
            pre_residuals=pre_residuals,
            confidence=float(spec.inference.confidence),
            null_value=null_value,
            alternative=_alternative_from_framework(spec),
        )
    except Exception as exc:
        fallback_warning = (
            f"Moving-block conformal inversion failed; split conformal fallback was used: {exc}"
        )
        inference = split_conformal_counterfactual_interval(
            post_observed,
            post_counterfactual,
            pre_observed=pre_observed,
            pre_counterfactual=pre_counterfactual,
            confidence=float(spec.inference.confidence),
            null_value=null_value,
            alternative=_alternative_from_framework(spec),
        )
    n_post = max(int(post_gaps.size), 1)
    scale = 1.0 / n_post if result.estimand_spec.time_aggregation == "post_period_average" else 1.0
    interval = (
        None
        if inference.interval is None
        else (float(inference.interval[0] * scale), float(inference.interval[1] * scale))
    )
    point_estimate = float(np.sum(post_gaps) * scale)
    exact_match = bool(
        np.isclose(
            point_estimate,
            float(result.estimate),
            rtol=1e-7,
            atol=max(1e-10, abs(float(result.estimate)) * 1e-9),
        )
    )
    scaled_null_distribution = dict(inference.null_distribution or {})
    for key in ("observed_statistic", "null_value"):
        value = _finite_number(scaled_null_distribution.get(key))
        if value is not None:
            scaled_null_distribution[key] = value * scale
    return replace(
        inference,
        interval=interval,
        estimand_spec=result.estimand_spec,
        point_estimate=point_estimate,
        primary_eligible=exact_match,
        standard_error=(
            None if inference.standard_error is None else float(inference.standard_error * scale)
        ),
        null_distribution=scaled_null_distribution,
        diagnostics={
            **inference.diagnostics,
            "source_estimator": result.estimator_name,
            "source_metric": result.metric,
            "source_cumulative_interval": inference.interval,
            "reported_scale_factor": scale,
            "estimand_contract_match": exact_match,
        },
        warnings=[
            *inference.warnings,
            *([] if fallback_warning is None else [fallback_warning]),
            *(
                []
                if exact_match
                else [
                    "Configured conformal inference was retained as supplementary because its "
                    "counterfactual-path statistic did not match the estimator point estimate."
                ]
            ),
        ],
    )


def _run_few_cluster_wild_bootstrap(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    result: EstimatorResult,
    spec: Any,
) -> InferenceResult:
    effects = _market_effect_frame(panel, design, metric)
    treatment = effects.loc[effects["treated"].astype(bool), "effect"].to_numpy(dtype=float)
    control = effects.loc[~effects["treated"].astype(bool), "effect"].to_numpy(dtype=float)
    n_treatment = len(design.treatment_geos)
    n_control = len(design.control_geos)
    if min(n_treatment, n_control) < 2:
        return InferenceResult(
            method="few_cluster_wild_bootstrap",
            method_family="small_sample",
            confidence=float(spec.inference.confidence),
            estimand_spec=_market_effect_estimand(metric, result.metric),
            primary_eligible=False,
            assumptions=[
                "Market-level wild bootstrap requires arm-level market variation.",
                "One-treated-market designs require assignment-aware randomization inference.",
            ],
            diagnostics={
                "source_estimator": result.estimator_name,
                "n_treatment_markets": n_treatment,
                "n_control_markets": n_control,
                "status": "not_evaluable_fewer_than_two_markets_per_arm",
            },
            warnings=[
                "Few-cluster wild bootstrap was not promoted because one arm has fewer than "
                "two markets; the treated-arm bootstrap distribution is degenerate in "
                "one-treated-geo designs."
            ],
        )
    df = max(n_treatment + n_control - 2, 1)
    observed = float(np.mean(treatment) - np.mean(control))
    rng = np.random.default_rng(_assignment_seed(spec))
    n_resamples = int(spec.inference.bootstrap_samples)
    treatment_centered = treatment - np.mean(treatment)
    control_centered = control - np.mean(control)
    se_observed = _welch_se(treatment, control)
    if se_observed is None or se_observed <= 0:
        raise ValueError("few_cluster_robust requires nonzero arm-level market variation")
    null_value = _null_value_from_framework(
        spec,
        metric,
        baseline=_effect_frame_baseline(effects),
    )
    t_observed = (observed - null_value) / se_observed
    draws = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        t_weights = rng.choice([-1.0, 1.0], size=treatment_centered.size)
        c_weights = rng.choice([-1.0, 1.0], size=control_centered.size)
        t_draw = treatment_centered * t_weights
        c_draw = control_centered * c_weights
        se_draw = _welch_se(t_draw, c_draw)
        draws[index] = (
            float((np.mean(t_draw) - np.mean(c_draw)) / se_draw)
            if se_draw is not None and se_draw > 0
            else np.nan
        )
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        raise ValueError("few_cluster_robust produced no finite studentized draws")
    alpha = 1.0 - float(spec.inference.confidence)
    alternative = _alternative_from_framework(spec)
    if alternative == "greater":
        bootstrap_critical = float(np.quantile(draws, 1.0 - alpha))
        lower = float(observed - bootstrap_critical * se_observed)
        upper = float("inf")
        p_value = float((np.sum(draws >= t_observed - 1e-12) + 1) / (draws.size + 1))
    elif alternative == "less":
        bootstrap_critical = float(np.quantile(draws, alpha))
        lower = float("-inf")
        upper = float(observed - bootstrap_critical * se_observed)
        p_value = float((np.sum(draws <= t_observed + 1e-12) + 1) / (draws.size + 1))
    else:
        bootstrap_critical = float(np.quantile(np.abs(draws), 1.0 - alpha))
        lower = float(observed - bootstrap_critical * se_observed)
        upper = float(observed + bootstrap_critical * se_observed)
        p_value = float((np.sum(np.abs(draws) >= abs(t_observed) - 1e-12) + 1) / (draws.size + 1))
    interval = (lower, upper)
    if result.standard_error is not None and np.isfinite(result.standard_error):
        se = float(result.standard_error)
        if se > 0:
            statistic = float(result.estimate) / se
            t_reference_critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
            t_interval = (
                float(result.estimate - t_reference_critical * se),
                float(result.estimate + t_reference_critical * se),
            )
        else:
            statistic = None
            t_interval = None
    else:
        statistic = None
        t_interval = None
    warnings = [
        (
            "Few-cluster robust inference uses a market-level wild bootstrap on "
            "pre/post market effects. Prefer assignment-aware randomization inference "
            "when the design assignment mechanism is known."
        )
    ]
    if min(n_treatment, n_control) < 3:
        warnings.append(
            "One arm has fewer than three markets; asymptotic cluster corrections are fragile."
        )
    return InferenceResult(
        method="few_cluster_wild_bootstrap",
        method_family="small_sample",
        interval=interval,
        interval_type="wild_cluster_bootstrap_t",
        p_value=p_value,
        confidence=float(spec.inference.confidence),
        standard_error=se_observed,
        estimand_spec=_market_effect_estimand(metric, result.metric),
        point_estimate=observed,
        primary_eligible=True,
        assumptions=[
            "Market-level pre/post effects are exchangeable within treatment arms.",
            "Rademacher wild weights approximate the small-sample null distribution.",
        ],
        diagnostics={
            "source_estimator": result.estimator_name,
            "n_treatment_markets": n_treatment,
            "n_control_markets": n_control,
            "degrees_of_freedom": df,
            "n_resamples": n_resamples,
            "observed_statistic": observed,
            "null_value": null_value,
            "studentized_statistic": t_observed,
            "studentized_source_statistic": statistic,
            "bootstrap_t_critical_value": bootstrap_critical,
            "t_reference_interval": t_interval,
            "alternative": alternative,
        },
        artifacts={
            "null_distribution": {
                "sample": draws[: min(len(draws), 500)].tolist(),
                "truncated": len(draws) > 500,
            }
        },
        warnings=warnings,
    )


def _run_monitoring_if_requested(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    spec: Any,
) -> InferenceResult | None:
    mode = spec.monitoring.mode
    if mode == "fixed_horizon":
        return None
    observations = _monitoring_effect_observations(panel, design, metric, spec)
    values = observations["values"]
    if values.size == 0:
        raise ValueError("monitoring was requested, but no post-period observations exist")
    if mode == "descriptive":
        return InferenceResult(
            method="descriptive_planned_looks",
            method_family="monitoring",
            primary_eligible=False,
            null_distribution={
                "observed_statistic": float(np.mean(values)),
                "n_looks": int(values.size),
            },
            diagnostics=observations["diagnostics"],
            artifacts=observations["artifacts"],
            warnings=observations["warnings"],
        )
    lower, upper, bound_diagnostics, bound_warnings = _monitoring_bounds(observations, spec)
    framework_null = _null_value_from_framework(
        spec,
        metric,
        baseline=observations.get("pre_treatment_mean"),
    )
    inference = bounded_mean_confidence_sequence(
        values,
        lower_bound=lower,
        upper_bound=upper,
        alpha=1.0 - float(spec.inference.confidence),
        null_value=framework_null if lower <= framework_null <= upper else None,
        alternative=_alternative_from_framework(spec),
        look_indexes=observations["look_indexes"],
    )
    method = (
        "planned_look_confidence_sequence"
        if mode == "planned_looks"
        else "anytime_valid_confidence_sequence"
    )
    interval_type = (
        "descriptive_bounded_mean_sequence"
        if bound_diagnostics.get("monitoring_bounds_expanded_after_observation")
        else inference.interval_type
    )
    method_family = (
        "monitoring"
        if bound_diagnostics.get("monitoring_bounds_expanded_after_observation")
        else inference.method_family
    )
    return replace(
        inference,
        method=method,
        method_family=method_family,
        primary_eligible=False,
        interval_type=interval_type,
        p_value=(
            None
            if bound_diagnostics.get("monitoring_bounds_expanded_after_observation")
            else inference.p_value
        ),
        diagnostics={
            **inference.diagnostics,
            **observations["diagnostics"],
            **bound_diagnostics,
            "monitoring_mode": mode,
        },
        artifacts={**inference.artifacts, **observations["artifacts"]},
        warnings=[*inference.warnings, *observations["warnings"], *bound_warnings],
    )


def _run_calibration_if_requested(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    estimator: Estimator,
    spec: Any,
) -> list[CalibrationResult]:
    if not _calibration_requested(spec):
        return []
    unsupported_shapes = [shape for shape in spec.calibration.effect_shapes if shape != "constant"]
    if unsupported_shapes and spec.calibration.injected_lifts:
        raise ValueError(
            "Injected-lift calibration currently supports constant effect_shapes only; "
            f"unsupported shapes: {', '.join(unsupported_shapes)}."
        )

    alpha = float(spec.calibration.target_false_positive_rate)
    results: list[CalibrationResult] = []
    if spec.calibration.placebo_windows > 0:
        estimator_name = getattr(estimator, "name", estimator.__class__.__name__)
        time_applicability = placebo_applicability(estimator_name, PLACEBO_IN_TIME)
        if time_applicability["applicable"]:
            results.append(
                placebo_backtest(
                    panel,
                    design,
                    metric,
                    estimator,
                    n_windows=int(spec.calibration.placebo_windows),
                    alpha=alpha,
                )
            )
        else:
            results.append(
                not_applicable_placebo_result(
                    estimator_name,
                    metric,
                    method=PLACEBO_IN_TIME,
                    reason=str(time_applicability["reason"]),
                )
            )

        space_applicability = placebo_applicability(estimator_name, PLACEBO_IN_SPACE)
        if space_applicability["applicable"]:
            results.append(
                placebo_in_space_backtest(
                    panel,
                    design,
                    metric,
                    estimator,
                    alpha=alpha,
                )
            )
        else:
            results.append(
                not_applicable_placebo_result(
                    estimator_name,
                    metric,
                    method=PLACEBO_IN_SPACE,
                    reason=str(space_applicability["reason"]),
                )
            )
    if spec.calibration.injected_lifts:
        results.append(
            injected_lift_recovery_curve(
                panel,
                design,
                metric,
                estimator,
                lifts=[float(lift) for lift in spec.calibration.injected_lifts],
            )
        )
    return results


def _market_effect_frame(panel: Any, design: CompletedDesign, metric: Any) -> pd.DataFrame:
    frame, info, _ = prepare_estimator_frame(panel, design, metric)
    grouped = (
        frame.groupby([design.geo_col, PERIOD_COL], observed=True)[OUTCOME_COL]
        .mean()
        .unstack(PERIOD_COL)
    )
    missing = [
        geo
        for geo in design.all_geos
        if geo not in grouped.index or {"pre", "post"}.difference(grouped.columns)
    ]
    if missing:
        raise ValueError(
            "Inference requires every design market to have pre and post data; "
            f"missing complete periods for {missing[:5]}."
        )
    rows = []
    for geo in design.all_geos:
        effect = float(grouped.loc[geo, "post"] - grouped.loc[geo, "pre"])
        rows.append(
            {
                "geo_id": geo,
                "effect": effect,
                "pre": float(grouped.loc[geo, "pre"]),
                "treated": geo in design.treatment_geos,
                "metric": info.name,
                "metric_kind": info.kind,
            }
        )
    return pd.DataFrame(rows)


def _market_effect_estimand(metric: Any, metric_name: str) -> EstimandSpec:
    is_ratio = bool(
        getattr(metric, "numerator", None) is not None
        and getattr(metric, "denominator", None) is not None
    )
    return EstimandSpec(
        label="market_level_pre_post_difference_in_means",
        metric=metric_name,
        outcome_scale="unit_time_ratio_effect" if is_ratio else "absolute_effect",
        target_population="treated_markets",
        time_aggregation="post_period_average",
        population_aggregation="per_treated_market_average",
        causal_quantity="ATT",
        denominator_handling="mean_of_unit_time_ratios" if is_ratio else None,
        effect_unit="ratio_points" if is_ratio else "outcome_units",
        notes=(
            "Configured generic inference operates on market-level pre/post means and is "
            "primary only when that statistic exactly matches the estimator result."
        ),
    )


def _effect_frame_statistic(frame: pd.DataFrame) -> float:
    treatment = frame.loc[frame["treated"].astype(bool), "effect"].to_numpy(dtype=float)
    control = frame.loc[~frame["treated"].astype(bool), "effect"].to_numpy(dtype=float)
    if treatment.size == 0 or control.size == 0:
        raise ValueError("resampled statistic requires treatment and control markets")
    return float(np.mean(treatment) - np.mean(control))


def _monitoring_effect_observations(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    spec: Any,
) -> dict[str, Any]:
    frame, _, _ = prepare_estimator_frame(panel, design, metric)
    daily = (
        frame.groupby([design.time_col, PERIOD_COL, "ft_treated"], observed=True)[OUTCOME_COL]
        .mean()
        .reset_index()
    )
    pre = daily[daily[PERIOD_COL] == "pre"]
    post = daily[daily[PERIOD_COL] == "post"]
    pre_treatment = pre.loc[pre["ft_treated"] == 1, OUTCOME_COL]
    pre_control = pre.loc[pre["ft_treated"] == 0, OUTCOME_COL]
    if pre_treatment.empty or pre_control.empty:
        raise ValueError("monitoring requires treatment and control observations in the pre period")
    baseline_gap = float(pre_treatment.mean() - pre_control.mean())
    pre_rows = []
    for date_value, group in pre.groupby(design.time_col, observed=True):
        treated = group.loc[group["ft_treated"] == 1, OUTCOME_COL]
        control = group.loc[group["ft_treated"] == 0, OUTCOME_COL]
        if treated.empty or control.empty:
            continue
        value = float(treated.mean() - control.mean() - baseline_gap)
        pre_rows.append({"date": pd.Timestamp(date_value).date().isoformat(), "value": value})
    rows = []
    for date_value, group in post.groupby(design.time_col, observed=True):
        treated = group.loc[group["ft_treated"] == 1, OUTCOME_COL]
        control = group.loc[group["ft_treated"] == 0, OUTCOME_COL]
        if treated.empty or control.empty:
            continue
        value = float(treated.mean() - control.mean() - baseline_gap)
        rows.append(
            {
                "date": pd.Timestamp(date_value).date().isoformat(),
                "value": value,
                "look_index": len(rows) + 1,
            }
        )
    selected = _select_monitoring_rows(rows, spec.monitoring)
    max_look_index = max(int(row["look_index"]) for row in selected) if selected else 0
    accumulated_rows = rows[:max_look_index]
    values = np.asarray([row["value"] for row in accumulated_rows], dtype=float)
    look_indexes = [int(row["look_index"]) for row in selected]
    warnings = []
    if not spec.monitoring.look_dates and not spec.monitoring.information_fractions:
        warnings.append(
            "Monitoring used every post-period date because no look_dates or "
            "information_fractions were configured."
        )
    return {
        "values": values,
        "bound_values": np.asarray([row["value"] for row in pre_rows], dtype=float),
        "pre_outcome_min": float(pre[OUTCOME_COL].min()),
        "pre_outcome_max": float(pre[OUTCOME_COL].max()),
        "pre_treatment_mean": float(pre_treatment.mean()),
        "diagnostics": {
            "n_post_dates": len(rows),
            "n_pre_bound_observations": len(pre_rows),
            "n_monitoring_observations": int(len(selected)),
            "n_accumulated_observations": int(values.size),
            "baseline_pre_gap": baseline_gap,
            "look_indexes": look_indexes,
            "look_dates": list(spec.monitoring.look_dates),
            "information_fractions": list(spec.monitoring.information_fractions),
        },
        "artifacts": {
            "monitoring_observations": selected,
            "monitoring_accumulated_observations": accumulated_rows,
            "pre_monitoring_bound_observations": pre_rows,
        },
        "look_indexes": look_indexes,
        "warnings": warnings,
    }


def _select_monitoring_rows(
    rows: list[dict[str, Any]],
    monitoring_spec: Any,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    if monitoring_spec.look_dates:
        look_dates = [
            pd.Timestamp(value).date().isoformat() for value in monitoring_spec.look_dates
        ]
        selected = [row for row in rows if row["date"] in set(look_dates)]
        if selected:
            return selected
    if monitoring_spec.information_fractions:
        indexes = sorted(
            {
                min(max(int(np.ceil(float(fraction) * len(rows))) - 1, 0), len(rows) - 1)
                for fraction in monitoring_spec.information_fractions
            }
        )
        return [rows[index] for index in indexes]
    return rows


def _monitoring_bounds(
    observations: dict[str, Any],
    spec: Any,
) -> tuple[float, float, dict[str, Any], list[str]]:
    values = np.asarray(observations["values"], dtype=float)
    configured_lower = getattr(spec.monitoring, "lower_bound", None)
    configured_upper = getattr(spec.monitoring, "upper_bound", None)
    warnings: list[str] = []
    diagnostics: dict[str, Any] = {
        "monitoring_bound_source": "configured",
        "monitoring_bounds_expanded_after_observation": False,
    }
    if configured_lower is not None and configured_upper is not None:
        lower = float(configured_lower)
        upper = float(configured_upper)
    else:
        lower, upper = _bounds_for_observations(
            np.asarray(observations["bound_values"], dtype=float),
            support=(observations["pre_outcome_min"], observations["pre_outcome_max"]),
            center_shift=float(observations["diagnostics"]["baseline_pre_gap"]),
        )
        diagnostics["monitoring_bound_source"] = "pre_period_history"
    if np.any(values < lower) or np.any(values > upper):
        diagnostics["monitoring_bounds_expanded_after_observation"] = True
        diagnostics["monitoring_pre_expansion_bounds"] = [lower, upper]
        warnings.append(
            "Monitoring observations exceeded the configured or pre-period-derived bounded "
            "range. Bounds were widened so the sequence can be reported, but strict "
            "anytime-valid guarantees require pre-specified bounds that contain all looks."
        )
        lower = min(lower, float(np.min(values)))
        upper = max(upper, float(np.max(values)))
        padding = max((upper - lower) * 0.01, 1e-9)
        lower -= padding
        upper += padding
    diagnostics["monitoring_lower_bound"] = lower
    diagnostics["monitoring_upper_bound"] = upper
    return lower, upper, diagnostics, warnings


def _bounds_for_observations(
    values: np.ndarray,
    *,
    support: tuple[float, float] | None = None,
    center_shift: float = 0.0,
) -> tuple[float, float]:
    low = float(min(np.min(values), 0.0))
    high = float(max(np.max(values), 0.0))
    if support is not None:
        support_low, support_high = support
        low = min(low, float(support_low - support_high - center_shift))
        high = max(high, float(support_high - support_low - center_shift))
    if np.isclose(low, high):
        return (low - 1.0, high + 1.0)
    padding = max((high - low) * 0.1, 1e-9)
    return low - padding, high + padding


def _welch_se(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    variance = float(np.var(left, ddof=1) / left.size + np.var(right, ddof=1) / right.size)
    if variance <= 0 or not np.isfinite(variance):
        return None
    return float(np.sqrt(variance))


def _assignment_policy_from_completed_spec(
    spec: Any,
    design: CompletedDesign,
) -> AssignmentPolicy | None:
    policy_spec = spec.assignment_policy
    if policy_spec is None:
        return None
    if policy_spec.kind not in {"fixed_treatment_count", "candidate_constrained"}:
        raise ValueError(
            "Completed-test randomization inference can reconstruct fixed_treatment_count "
            "and candidate_constrained assignment policies. Stratified, matched_pairs, and "
            "supergeo policies require explicit market-level assignment metadata."
        )
    # Roadmap-level constraint lists may reference markets the completed design never
    # used (e.g. forbidden markets correctly excluded at design time); those constraints
    # are vacuous over the design universe, so drop them instead of crashing.
    universe = {str(geo) for geo in design.all_geos}

    def in_universe(markets: Any) -> tuple[str, ...]:
        return tuple(market for market in map(str, markets) if market in universe)

    missing_required = sorted(
        set(map(str, policy_spec.required_treatment_markets)).difference(universe)
    )
    if missing_required:
        raise ValueError(
            "Completed design does not contain required treatment markets "
            f"{missing_required} declared by its assignment policy; the spec and the "
            "realized design disagree."
        )
    return AssignmentPolicy(
        markets=tuple(design.all_geos),
        treatment_count=int(policy_spec.treatment_count or len(design.treatment_geos)),
        kind=policy_spec.kind,
        required_treatment_markets=tuple(map(str, policy_spec.required_treatment_markets)),
        forbidden_treatment_markets=in_universe(policy_spec.forbidden_treatment_markets),
        fixed_control_markets=in_universe(policy_spec.fixed_control_markets),
        shared_control_markets=in_universe(policy_spec.shared_control_markets),
        seed=policy_spec.seed,
    )


def _assignment_seed(spec: Any) -> int | None:
    if spec.assignment_policy is not None:
        return spec.assignment_policy.seed
    return 0


def _max_exact_assignments(spec: Any) -> int:
    if spec.assignment_policy is not None:
        return int(spec.assignment_policy.max_enumerated_assignments)
    return 100_000


def _alternative_from_framework(spec: Any) -> str:
    kind = str(getattr(spec.test_framework, "kind", "two_sided"))
    if kind in {"superiority", "non_inferiority"}:
        return "greater"
    if kind == "inferiority":
        return "less"
    return "two-sided"


def _null_value_from_framework(spec: Any, metric: Any, *, baseline: float | None = None) -> float:
    """Return the framework-implied null value on the statistic's estimate scale.

    Non-inferiority pairs with alternative='greater' and tests H0: effect <= -margin;
    inferiority pairs with alternative='less' and tests H0: effect >= margin. Margins
    declared on the relative-lift scale are converted with ``baseline`` (the relative
    lift denominator on the statistic's scale); without a finite nonzero baseline the
    margin cannot be converted and the null stays at zero.
    """

    framework = getattr(spec, "test_framework", None)
    if framework is None:
        return 0.0
    kind = str(getattr(framework, "kind", "two_sided"))
    if kind == "non_inferiority":
        sign = -1.0
    elif kind == "inferiority":
        sign = 1.0
    else:
        return 0.0
    margins = getattr(framework, "margins", None) or {}
    metric_name = str(getattr(metric, "name", metric))
    raw = margins.get(metric_name)
    if raw is None:
        raw = getattr(framework, "default_margin", 0.0)
    margin = abs(_finite_number(raw) or 0.0)
    if margin == 0.0:
        return 0.0
    if str(getattr(framework, "effect_scale", "estimate")) == "relative_lift":
        baseline_value = _finite_number(baseline)
        if baseline_value is None or baseline_value == 0.0:
            return 0.0
        margin *= abs(baseline_value)
    return sign * margin


def _effect_frame_baseline(effects: pd.DataFrame) -> float | None:
    """Treatment-arm pre-period mean, the relative-lift denominator for market effects."""

    if "pre" not in effects.columns:
        return None
    treated = effects.loc[effects["treated"].astype(bool), "pre"].to_numpy(dtype=float)
    if treated.size == 0:
        return None
    return _finite_number(np.mean(treated))


def _counterfactual_series(
    records: Sequence[dict[str, Any]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    pre_observed: list[float] = []
    pre_counterfactual: list[float] = []
    post_observed: list[float] = []
    post_counterfactual: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        observed = _finite(record.get("observed"))
        counterfactual = _finite(
            record.get("counterfactual")
            if record.get("counterfactual") is not None
            else record.get("augmented_counterfactual")
        )
        if observed is None or counterfactual is None:
            continue
        period = str(record.get("period") or "").lower()
        if period == "pre":
            pre_observed.append(observed)
            pre_counterfactual.append(counterfactual)
        elif period == "post" or not period:
            post_observed.append(observed)
            post_counterfactual.append(counterfactual)
    if len(pre_observed) < 2 or not post_observed:
        raise ValueError(
            "conformal_inference requires at least two pre-period counterfactual residuals "
            "and at least one post-period counterfactual observation."
        )
    return pre_observed, pre_counterfactual, post_observed, post_counterfactual


def _primary_p_value(result: EstimatorResult) -> float | None:
    candidates = [
        inference.p_value
        for inference in result.inference_results
        if inference.diagnostics.get("selected_as_primary")
    ]
    candidates.append(result.p_value)
    candidates.extend(inference.p_value for inference in result.inference_results)
    for value in candidates:
        number = _finite(value)
        if number is not None and 0.0 <= number <= 1.0:
            return number
    return None


def _calibration_requested(spec: Any) -> bool:
    return bool(spec.calibration.placebo_windows > 0 or spec.calibration.injected_lifts)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _unique(values: Any) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output
