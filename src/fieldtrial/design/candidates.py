"""Candidate generation for treatment/control market sets."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from fieldtrial.data.panel import GeoPanel
from fieldtrial.design.interference import MarketGraph, graph_from_interference_spec
from fieldtrial.design.matching import construct_matched_pairs, strata_values_from_columns
from fieldtrial.design.policies import AssignmentPolicy, FeasibleAssignment
from fieldtrial.design.specs import ExperimentSpec, RoadmapDefaults, RoadmapSpec
from fieldtrial.design.supergeo import Supergeo, build_supergeos, expand_supergeo_units
from fieldtrial.metrics.base import MetricSpec
from fieldtrial.metrics.catalog import MetricCatalog
from fieldtrial.metrics.ratio import RatioMetric
from fieldtrial.power.mde import approximate_count_mde, ratio_delta_mde


class MDEComputationError(ValueError):
    """Raised when a candidate's MDE cannot be computed from observed data."""


@dataclass(frozen=True)
class CandidateDesign:
    candidate_id: str
    test_name: str
    start_date: date
    end_date: date
    duration_days: int
    treatment_markets: list[str]
    control_markets: list[str]
    metric_mde: dict[str, float]
    objective_score: float
    score_components: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    metric_roles: dict[str, str] = field(default_factory=dict)
    market_profile: dict[str, Any] = field(default_factory=dict)
    test_framework: dict[str, Any] = field(default_factory=dict)
    assignment_policy: dict[str, Any] = field(default_factory=dict)
    balance_diagnostics: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    method_readiness: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["start_date"] = self.start_date.isoformat()
        payload["end_date"] = self.end_date.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CandidateDesign:
        data = dict(payload)
        data["start_date"] = pd.Timestamp(data["start_date"]).date()
        data["end_date"] = pd.Timestamp(data["end_date"]).date()
        return cls(**data)


@dataclass(frozen=True)
class _PolicyContext:
    spec_payload: dict[str, Any]
    policy: AssignmentPolicy
    unit_to_markets: dict[str, list[str]]
    unit_type: str = "market"
    supergeos: tuple[Supergeo, ...] = ()
    warnings: tuple[str, ...] = ()
    interference_graph: MarketGraph | None = None
    interference_warnings: tuple[str, ...] = ()
    interference_spec: dict[str, Any] = field(default_factory=dict)


class CandidateGenerator:
    def __init__(
        self,
        panel: GeoPanel,
        roadmap: RoadmapSpec,
        *,
        registry: Any | None = None,
        seed: int = 123,
    ) -> None:
        self.panel = panel
        self.roadmap = roadmap
        self.registry = registry
        self.seed = seed

    def generate(self, *, max_per_test: int | None = None) -> dict[str, list[CandidateDesign]]:
        return {
            test.name: self.generate_for_test(
                test, max_candidates=max_per_test or self.roadmap.defaults.candidate_count
            )
            for test in self.roadmap.tests
        }

    def generate_for_test(
        self,
        spec: ExperimentSpec,
        *,
        max_candidates: int = 50,
        defaults: RoadmapDefaults | None = None,
    ) -> list[CandidateDesign]:
        defaults = defaults or self.roadmap.defaults
        digest = hashlib.sha256(f"{self.seed}:{spec.name}".encode()).hexdigest()
        rng = np.random.default_rng(int(digest[:8], 16))
        markets = self._eligible_markets(spec)
        if len(markets) < 3:
            return []
        volume = self._market_volume(spec, markets)
        starts = self._candidate_starts(spec)
        candidates: list[CandidateDesign] = []
        min_t = max(1, int(np.ceil(len(markets) * spec.effective_min_treatment_share(defaults))))
        max_t = max(
            min_t, int(np.floor(len(markets) * spec.effective_max_treatment_share(defaults)))
        )
        min_controls = spec.effective_min_control_markets(defaults)
        policy_context = self._policy_context(
            spec,
            defaults=defaults,
            markets=markets,
            default_treatment_count=max(min_t, min(max_t, int(round((min_t + max_t) / 2)))),
        )
        if policy_context.policy.treatment_count >= len(policy_context.policy.markets):
            return []

        attempts = 0
        seen: set[tuple[date, int, tuple[str, ...]]] = set()
        while len(candidates) < max_candidates and attempts < max_candidates * 100:
            attempts += 1
            duration = int(rng.choice(spec.candidate_durations))
            start = starts[int(rng.integers(0, len(starts)))] if starts else spec.earliest_start
            end = start + timedelta(days=duration - 1)
            if end > spec.latest_end or any(
                window.overlaps(start, end) for window in spec.blackout_windows
            ):
                continue
            assignment = self._draw_policy_assignment(policy_context, rng)
            treatment = self._expand_assignment_units(policy_context, assignment.treatment_markets)
            key = (start, duration, tuple(treatment))
            if key in seen:
                continue
            seen.add(key)
            blocks = self._registry_blocks(start, end)
            if set(treatment).intersection(blocks["blocked_from_treatment"]):
                continue
            controls, control_warnings, interference_diagnostics = self._controls_from_assignment(
                policy_context,
                assignment=assignment,
                treatment=treatment,
                blocked=blocks["blocked_from_control"],
                volume=volume,
                min_controls=min_controls,
                rng=rng,
            )
            if len(controls) < min_controls:
                continue
            balance = self._balance_diagnostics(
                spec,
                treatment,
                controls,
                start,
            )
            if not self._passes_rerandomization(
                balance,
                spec.effective_assignment_policy(defaults).rerandomization,
            ):
                continue
            metric_mde = self._score_mde(spec, treatment, controls, start, duration_days=duration)
            score, score_components = self._candidate_score(
                spec,
                metric_mde,
                control_count=len(controls),
            )
            assignment_policy = policy_context.spec_payload
            policy_warnings = [
                *policy_context.warnings,
                *policy_context.interference_warnings,
                *control_warnings,
            ]
            candidate_id = f"{spec.name}:{len(candidates):04d}"
            candidates.append(
                CandidateDesign(
                    candidate_id=candidate_id,
                    test_name=spec.name,
                    start_date=start,
                    end_date=end,
                    duration_days=duration,
                    treatment_markets=treatment,
                    control_markets=controls,
                    metric_mde=metric_mde,
                    objective_score=score,
                    score_components=score_components,
                    warnings=[
                        *([] if len(controls) >= min_controls * 2 else ["small_control_pool"]),
                        *policy_warnings,
                    ],
                    metric_roles={
                        name: getattr(config.role, "value", str(config.role))
                        for name, config in spec.metrics.items()
                    },
                    market_profile=self._market_profile(treatment, controls),
                    test_framework=spec.test_framework.model_dump(mode="json"),
                    assignment_policy=assignment_policy,
                    balance_diagnostics=balance,
                    calibration=spec.effective_calibration(defaults).model_dump(mode="json"),
                    method_readiness={
                        "estimator_suite": spec.effective_estimator_suite(defaults).model_dump(
                            mode="json"
                        ),
                        "inference": spec.effective_inference(defaults).model_dump(mode="json"),
                        "monitoring": spec.effective_monitoring(defaults).model_dump(mode="json"),
                        "interference": spec.effective_interference(defaults).model_dump(
                            mode="json"
                        ),
                        "assignment_policy_execution": {
                            "status": "executed",
                            "unit_type": policy_context.unit_type,
                            "assignment_metadata": assignment.metadata,
                            "warnings": policy_warnings,
                        },
                        "interference_diagnostics": interference_diagnostics,
                    },
                    metadata={
                        "domain": getattr(spec.domain, "value", str(spec.domain)),
                        "required": spec.required,
                        "tags": list(spec.tags),
                    },
                )
            )
        return candidates

    @staticmethod
    def _assignment_policy_warnings(policy: Any) -> list[str]:
        kind = str(getattr(policy, "kind", "fixed_treatment_count"))
        if kind in {
            "fixed_treatment_count",
            "candidate_constrained",
            "stratified",
            "matched_pairs",
            "supergeo",
        }:
            return []
        return [f"unsupported_assignment_policy_kind: {kind}"]

    def _policy_context(
        self,
        spec: ExperimentSpec,
        *,
        defaults: RoadmapDefaults,
        markets: list[str],
        default_treatment_count: int,
    ) -> _PolicyContext:
        policy_spec = spec.effective_assignment_policy(defaults)
        warnings = self._assignment_policy_warnings(policy_spec)
        kind = str(policy_spec.kind)
        interference = spec.effective_interference(defaults)
        graph, interference_warnings = graph_from_interference_spec(
            interference,
            self.panel,
            markets,
        )

        if kind == "supergeo":
            supergeos = tuple(
                build_supergeos(
                    self.panel,
                    markets,
                    min_volume=policy_spec.min_supergeo_volume,
                    max_markets_per_group=policy_spec.max_markets_per_supergeo,
                    volume_column=policy_spec.supergeo_volume_column,
                    group_columns=policy_spec.supergeo_group_columns,
                )
            )
            unit_ids = [unit.supergeo_id for unit in supergeos]
            unit_to_markets = {unit.supergeo_id: list(unit.markets) for unit in supergeos}
            treatment_count = min(
                int(policy_spec.treatment_count or default_treatment_count),
                max(len(unit_ids) - 1, 1),
            )
            policy = self._assignment_policy_from_spec(
                policy_spec,
                unit_ids,
                treatment_count=treatment_count,
                unit_to_markets=unit_to_markets,
            )
            payload = policy_spec.model_dump(mode="json")
            payload["execution"] = {
                "status": "executed",
                "unit_type": "supergeo",
                "supergeos": [unit.to_dict() for unit in supergeos],
            }
            return _PolicyContext(
                spec_payload=payload,
                policy=policy,
                unit_to_markets=unit_to_markets,
                unit_type="supergeo",
                supergeos=supergeos,
                warnings=tuple(warnings),
                interference_graph=graph,
                interference_warnings=tuple(interference_warnings),
                interference_spec=interference.model_dump(mode="json"),
            )

        pairs = None
        unit_markets = markets
        if kind == "matched_pairs":
            matching_end = spec.earliest_start - timedelta(days=1)
            pairs = construct_matched_pairs(
                self.panel,
                markets,
                n_pairs=policy_spec.treatment_count or default_treatment_count,
                end=matching_end,
                metric_columns=policy_spec.matching_metrics,
                exact_match_columns=policy_spec.matching_columns or policy_spec.strata,
                max_distance=policy_spec.max_pair_distance,
            )
            paired_markets = sorted({market for pair in pairs for market in pair.markets})
            if not policy_spec.allow_unpaired_markets:
                unit_markets = paired_markets
            treatment_count = len(pairs)
        else:
            treatment_count = int(policy_spec.treatment_count or default_treatment_count)

        strata = (
            strata_values_from_columns(self.panel, unit_markets, policy_spec.strata)
            if kind == "stratified"
            else {}
        )
        if kind == "stratified" and policy_spec.strata and not strata:
            warnings.append("stratified_assignment_requested_but_no_strata_columns_found")
        policy = self._assignment_policy_from_spec(
            policy_spec,
            unit_markets,
            treatment_count=treatment_count,
            strata_values=strata,
            pairs=[pair.markets for pair in pairs or []],
        )
        payload = policy_spec.model_dump(mode="json")
        payload["execution"] = {
            "status": "executed",
            "unit_type": "market",
            "strata_values": strata,
            "matched_pairs": [pair.to_dict() for pair in pairs or []],
            "matching_window": (
                {"end": matching_end.isoformat(), "source": "before_earliest_start"}
                if kind == "matched_pairs"
                else None
            ),
        }
        return _PolicyContext(
            spec_payload=payload,
            policy=policy,
            unit_to_markets={market: [market] for market in unit_markets},
            warnings=tuple(warnings),
            interference_graph=graph,
            interference_warnings=tuple(interference_warnings),
            interference_spec=interference.model_dump(mode="json"),
        )

    def _assignment_policy_from_spec(
        self,
        policy_spec: Any,
        units: list[str] | tuple[str, ...],
        *,
        treatment_count: int,
        unit_to_markets: dict[str, list[str]] | None = None,
        strata_values: dict[str, str] | None = None,
        pairs: list[tuple[str, str]] | None = None,
    ) -> AssignmentPolicy:
        unit_tuple = tuple(str(unit) for unit in units)
        unit_to_markets = unit_to_markets or {unit: [unit] for unit in unit_tuple}
        reverse_lookup: dict[str, str] = {}
        for unit, unit_markets in unit_to_markets.items():
            for market in unit_markets:
                reverse_lookup[str(market)] = str(unit)

        def map_units(markets: list[str]) -> tuple[str, ...]:
            mapped = [reverse_lookup.get(str(market), str(market)) for market in markets]
            return tuple(dict.fromkeys(unit for unit in mapped if unit in unit_tuple))

        return AssignmentPolicy(
            markets=unit_tuple,
            treatment_count=max(1, int(treatment_count)),
            kind=str(policy_spec.kind),
            strata=strata_values or {},
            pairs=tuple((str(a), str(b)) for a, b in (pairs or [])),
            required_treatment_markets=map_units(policy_spec.required_treatment_markets),
            forbidden_treatment_markets=map_units(policy_spec.forbidden_treatment_markets),
            fixed_control_markets=map_units(policy_spec.fixed_control_markets),
            shared_control_markets=map_units(policy_spec.shared_control_markets),
            seed=policy_spec.seed,
        )

    def _draw_policy_assignment(
        self,
        context: _PolicyContext,
        rng: np.random.Generator,
    ) -> FeasibleAssignment:
        if context.policy.n_feasible_assignments <= context.policy.treatment_count * 20:
            assignments = context.policy.enumerate(
                max_assignments=max(context.policy.n_feasible_assignments, 1)
            )
            if not assignments:
                raise ValueError(
                    "assignment policy has no feasible assignments given its "
                    "required/forbidden/fixed-control constraints"
                )
            return assignments[int(rng.integers(0, len(assignments)))]
        return context.policy.sample(1, seed=int(rng.integers(0, np.iinfo(np.int32).max)))[0]

    @staticmethod
    def _expand_assignment_units(
        context: _PolicyContext,
        units: tuple[str, ...],
    ) -> list[str]:
        if context.unit_type == "supergeo":
            return expand_supergeo_units(context.supergeos, units)
        markets: list[str] = []
        for unit in units:
            markets.extend(context.unit_to_markets.get(unit, [unit]))
        return sorted(dict.fromkeys(markets))

    def _controls_from_assignment(
        self,
        context: _PolicyContext,
        *,
        assignment: FeasibleAssignment,
        treatment: list[str],
        blocked: set[str],
        volume: pd.Series,
        min_controls: int,
        rng: np.random.Generator,
    ) -> tuple[list[str], list[str], dict[str, Any]]:
        controls = self._expand_assignment_units(context, assignment.control_markets)
        controls = [market for market in controls if market not in blocked]
        warnings: list[str] = []
        diagnostics: dict[str, Any] = {}
        graph = context.interference_graph
        if graph is not None:
            buffer_radius = context.interference_spec.get("buffer_radius")
            diagnostics = graph.contamination_score(
                treatment,
                controls,
                max_distance=buffer_radius,
            )
            contaminated = set(diagnostics.get("contaminated_controls") or [])
            if contaminated:
                warnings.append("control_spillover_risk_detected")
            if contaminated and context.interference_spec.get("exclude_buffer_controls"):
                fixed_controls = set(context.policy.fixed_control_markets)
                if fixed_controls.intersection(contaminated):
                    return [], ["fixed_control_spillover_conflict"], diagnostics
                controls = [market for market in controls if market not in contaminated]
                diagnostics = graph.contamination_score(
                    treatment,
                    controls,
                    max_distance=buffer_radius,
                )

        if context.policy.kind in {"matched_pairs", "supergeo", "stratified"}:
            return sorted(controls), warnings, diagnostics
        forced = [
            market
            for market in [
                *context.policy.fixed_control_markets,
                *context.policy.shared_control_markets,
            ]
            if market in controls
        ]
        if len(controls) <= max(min_controls * 3, min_controls):
            selected = sorted(controls)
        else:
            selected = self._select_controls(
                markets=controls,
                treatment=[],
                blocked=set(),
                volume=volume,
                min_controls=max(min_controls - len(forced), 1),
                rng=rng,
            )
            selected = sorted(dict.fromkeys([*forced, *selected]))
        return selected, warnings, diagnostics

    @staticmethod
    def _passes_rerandomization(
        balance: dict[str, Any],
        thresholds: dict[str, Any],
    ) -> bool:
        if not thresholds:
            return True
        smd_limit = thresholds.get("max_abs_smd") or thresholds.get(
            "max_standardized_mean_difference"
        )
        if smd_limit is not None:
            smd = balance.get("standardized_mean_difference")
            if smd is not None and abs(float(smd)) > float(smd_limit):
                return False
        trend_limit = thresholds.get("max_abs_trend_difference")
        if trend_limit is not None:
            trend = balance.get("pre_period_trend_difference")
            if trend is not None and abs(float(trend)) > float(trend_limit):
                return False
        min_vr = thresholds.get("min_variance_ratio")
        max_vr = thresholds.get("max_variance_ratio")
        variance_ratio = balance.get("variance_ratio")
        if variance_ratio is not None:
            if min_vr is not None and float(variance_ratio) < float(min_vr):
                return False
            if max_vr is not None and float(variance_ratio) > float(max_vr):
                return False
        return True

    def _eligible_markets(self, spec: ExperimentSpec) -> list[str]:
        if isinstance(spec.eligible_markets, list):
            markets = [str(m) for m in spec.eligible_markets]
        elif isinstance(self.roadmap.markets.universe, list):
            markets = [str(m) for m in self.roadmap.markets.universe]
        else:
            markets = self.panel.markets
        excluded = set(spec.excluded_markets).union(self.roadmap.markets.excluded)
        return [m for m in sorted(markets) if m not in excluded]

    def _candidate_starts(self, spec: ExperimentSpec) -> list[date]:
        latest_duration = max(int(d) for d in spec.candidate_durations)
        latest_start = spec.latest_end - timedelta(days=latest_duration - 1)
        starts = []
        current = spec.earliest_start
        while current <= latest_start:
            starts.append(current)
            current += timedelta(days=7)
        return starts or [spec.earliest_start]

    def _market_volume(self, spec: ExperimentSpec, markets: list[str]) -> pd.Series:
        first_metric = next(iter(spec.metrics.values()))
        col = getattr(first_metric, "column", None) or getattr(first_metric, "numerator", None)
        if col not in self.panel.df.columns:
            col = "orders" if "orders" in self.panel.df.columns else self.panel.metric_columns[0]
        return (
            self.panel.df[self.panel.df[self.panel.geo_col].isin(markets)]
            .groupby(self.panel.geo_col)[col]
            .sum()
        )

    def _select_controls(
        self,
        *,
        markets: list[str],
        treatment: list[str],
        blocked: set[str],
        volume: pd.Series,
        min_controls: int,
        rng: np.random.Generator,
    ) -> list[str]:
        treatment_set = set(treatment)
        eligible = [m for m in markets if m not in treatment_set and m not in blocked]
        if len(eligible) < min_controls:
            return []
        target_count = min(len(eligible), max(min_controls, min_controls * 3))
        fallback_volume = float(volume.mean()) if len(volume) else 0.0
        market_volume = volume.reindex(eligible).fillna(fallback_volume).astype(float)
        ordered = market_volume.sort_values(ascending=False).index.tolist()
        strata_count = max(1, min(4, target_count, len(ordered)))
        strata = [
            list(chunk) for chunk in np.array_split(np.array(ordered, dtype=object), strata_count)
        ]
        shuffled = [rng.permutation(stratum).tolist() for stratum in strata if stratum]

        selected: list[str] = []
        while len(selected) < target_count and any(shuffled):
            for stratum in shuffled:
                if not stratum:
                    continue
                selected.append(str(stratum.pop(0)))
                if len(selected) == target_count:
                    break

        return sorted(selected)

    def _registry_blocks(self, start: date, end: date) -> dict[str, set[str]]:
        if self.registry is None:
            return {
                "blocked_from_treatment": set(),
                "blocked_from_control": set(),
                "active_controls": set(),
            }
        active = self.registry.active_market_blocks(start_date=start, end_date=end)
        cooldown = self.registry.cooldown_blocks(start_date=start, end_date=end)
        active_treatment = set(active.get("blocked_from_treatment", active.get("treatment", set())))
        active_control = set(active.get("active_controls", active.get("control", set())))
        return {
            "blocked_from_treatment": active_treatment.union(cooldown),
            "blocked_from_control": active_treatment,
            "active_controls": active_control,
        }

    def _score_mde(
        self,
        spec: ExperimentSpec,
        treatment: list[str],
        controls: list[str],
        start: date,
        *,
        duration_days: int,
    ) -> dict[str, float]:
        pre = self.panel.df[self.panel.df[self.panel.time_col] < pd.Timestamp(start)]
        if pre.empty:
            pre = self.panel.df
        pre_period_days = int(pre[self.panel.time_col].nunique())
        catalog = MetricCatalog.from_configs(spec.metrics)
        power = spec.effective_power(self.roadmap.defaults)
        out: dict[str, float] = {}
        for metric_name in spec.metrics:
            metric = catalog.get(metric_name)
            try:
                if power.method == "estimator_replay":
                    value = self._replay_metric_mde(
                        pre,
                        metric,
                        spec=spec,
                        treatment=treatment,
                        controls=controls,
                        duration_days=duration_days,
                        power=power,
                    )
                elif isinstance(metric, RatioMetric):
                    value = ratio_delta_mde(
                        pre,
                        metric,
                        treatment_geos=treatment,
                        control_geos=controls,
                        test_length_days=duration_days,
                        pre_period_days=pre_period_days,
                        geo_col=self.panel.geo_col,
                        alpha=power.alpha,
                        power=power.target_power,
                    )
                else:
                    value = self._additive_metric_mde(
                        pre,
                        metric,
                        treatment=treatment,
                        controls=controls,
                        test_length_days=duration_days,
                        alpha=power.alpha,
                        power=power.target_power,
                    )
            except Exception as exc:
                raise MDEComputationError(
                    f"Could not compute MDE for metric {metric_name!r}: {exc}"
                ) from exc
            if not np.isfinite(value) or value < 0:
                raise MDEComputationError(
                    f"Could not compute MDE for metric {metric_name!r}: "
                    f"non-finite or negative value {value!r}"
                )
            out[metric_name] = float(value)
        return out

    def _replay_metric_mde(
        self,
        frame: pd.DataFrame,
        metric: MetricSpec,
        *,
        spec: ExperimentSpec,
        treatment: list[str],
        controls: list[str],
        duration_days: int,
        power: Any,
    ) -> float:
        # Imported lazily: the estimator registry pulls in the inference stack,
        # which the analytic planning path does not need.
        from fieldtrial.estimators.ensemble import instantiate_estimator
        from fieldtrial.power.replay import estimator_replay_power

        suite = spec.effective_estimator_suite(self.roadmap.defaults)
        estimator_name = power.replay_estimator or suite.estimators[0]
        estimator = instantiate_estimator(
            estimator_name,
            backend=suite.backend_overrides.get(estimator_name),
            params=suite.estimator_params.get(estimator_name),
        )
        result = estimator_replay_power(
            frame,
            metric,
            estimator,
            treatment_geos=treatment,
            control_geos=controls,
            duration_days=duration_days,
            lift_grid=power.lift_grid,
            alpha=power.alpha,
            target_power=power.target_power,
            n_windows=power.placebo_windows,
            geo_col=self.panel.geo_col,
            time_col=self.panel.time_col,
        )
        if result.evaluated_windows == 0:
            raise ValueError(
                "estimator replay evaluated no historical windows: "
                + "; ".join(result.errors or ["unknown reason"])
            )
        # When no grid lift reaches the power target the true MDE lies beyond
        # the grid; the largest grid lift is returned as a lower bound.
        return result.mde

    def _additive_metric_mde(
        self,
        frame: pd.DataFrame,
        metric: MetricSpec,
        *,
        treatment: list[str],
        controls: list[str],
        test_length_days: int,
        alpha: float,
        power: float,
    ) -> float:
        t = self._metric_daily_total(frame, metric, treatment)
        c = self._metric_daily_total(frame, metric, controls)
        if not t.index.equals(c.index):
            raise ValueError("treatment and control metric series have different dates")
        treatment_baseline = float(t.mean())
        control_baseline = float(c.mean())
        if not np.isfinite(treatment_baseline) or treatment_baseline <= 0:
            raise ValueError("relative MDE requires a positive treatment baseline")
        if not np.isfinite(control_baseline) or control_baseline <= 0:
            raise ValueError("relative MDE requires a positive control baseline")
        scaled_control = c * (treatment_baseline / control_baseline)
        return approximate_count_mde(
            t,
            scaled_control,
            test_length_days=test_length_days,
            alpha=alpha,
            power=power,
        )

    def _metric_daily_total(
        self,
        frame: pd.DataFrame,
        metric: MetricSpec,
        geos: list[str],
    ) -> pd.Series:
        if not geos:
            raise ValueError("MDE requires at least one market per arm")
        subset = frame[frame[self.panel.geo_col].isin(geos)]
        if subset.empty:
            raise ValueError("MDE inputs contain no rows for one arm")
        grouped = (
            subset.groupby([self.panel.geo_col, self.panel.time_col], observed=True)
            .apply(lambda group: metric.aggregate(group), include_groups=False)
            .rename("_metric_value")
            .reset_index()
        )
        daily = grouped.groupby(self.panel.time_col, observed=True)["_metric_value"].sum()
        daily = pd.to_numeric(daily.sort_index(), errors="coerce").astype(float)
        if daily.empty:
            raise ValueError("MDE inputs contain no daily metric totals")
        if not np.isfinite(daily.to_numpy(dtype=float)).all():
            raise ValueError("MDE inputs contain non-finite daily metric totals")
        return daily

    def _candidate_score(
        self,
        spec: ExperimentSpec,
        metric_mde: dict[str, float],
        *,
        control_count: int,
    ) -> tuple[float, dict[str, float]]:
        objective = spec.effective_objective(self.roadmap.defaults)
        priority_component = float(spec.priority * 100)
        if objective.mode == "max_priority":
            mde_component = 0.0
            weight_sum = 0.0
            objective_metric_count = 0
        else:
            if not metric_mde:
                raise ValueError("MDE-based objective requires at least one scored metric")
            weights = self._objective_metric_weights(spec, metric_mde)
            weight_sum = float(sum(weights.values()))
            if not weights:
                raise ValueError("MDE-based objective requires at least one scored metric")
            for name in weights:
                value = float(metric_mde[name])
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"MDE for objective metric {name!r} must be finite and >= 0")
            if objective.mode == "minimax_normalized_mde":
                mde_component = max(float(metric_mde[name]) for name in weights)
            else:
                weighted = [
                    float(metric_mde[name]) * float(weight) for name, weight in weights.items()
                ]
                mde_component = float(sum(weighted) / weight_sum) if weight_sum else 1.0
            objective_metric_count = len(weights)
        mde_penalty = -100.0 * float(objective.mde_penalty) * float(mde_component)
        control_pool_penalty = -float(objective.control_overuse_penalty) * max(control_count, 0)
        score = priority_component + mde_penalty + control_pool_penalty
        return score, {
            "priority": priority_component,
            "mde_component": float(mde_component),
            "mde_penalty": mde_penalty,
            "control_pool_penalty": control_pool_penalty,
            "objective_metric_count": float(objective_metric_count),
            "objective_weight_sum": float(weight_sum),
            "total": score,
        }

    def _objective_metric_weights(
        self,
        spec: ExperimentSpec,
        metric_mde: dict[str, float],
    ) -> dict[str, float]:
        objective = spec.effective_objective(self.roadmap.defaults)
        if objective.metric_weights:
            names = [name for name in spec.primary_metrics if name in metric_mde]
            names.extend(name for name in objective.metric_weights if name in metric_mde)
            requested = set(spec.primary_metrics).union(objective.metric_weights)
        else:
            names = [name for name in spec.primary_metrics if name in metric_mde]
            requested = set(spec.primary_metrics)
        missing = sorted(name for name in requested if name not in metric_mde)
        if missing:
            raise ValueError(f"missing MDE for objective metric(s): {', '.join(missing)}")

        weights: dict[str, float] = {}
        for name in dict.fromkeys(names):
            weight = float(objective.metric_weights.get(name, 1.0))
            if weight > 0:
                weights[name] = weight
        if weights:
            return weights
        return {name: 1.0 for name in metric_mde}

    def _market_profile(self, treatment: list[str], controls: list[str]) -> dict[str, Any]:
        frame = self.panel.df
        geo_col = self.panel.geo_col
        profile: dict[str, Any] = {}
        for role, geos in [("treatment", treatment), ("control", controls)]:
            subset = frame[frame[geo_col].isin(geos)]
            market_rows = subset.drop_duplicates(geo_col)
            role_profile: dict[str, Any] = {"count": len(geos)}
            if "region" in market_rows.columns:
                role_profile["region_counts"] = {
                    str(key): int(value)
                    for key, value in market_rows["region"].value_counts().sort_index().items()
                }
            for size_col in ["market_weight", "market_size", "population"]:
                if size_col in market_rows.columns:
                    values = pd.to_numeric(market_rows[size_col], errors="coerce").dropna()
                    if not values.empty:
                        role_profile[f"{size_col}_median"] = float(values.median())
                        role_profile[f"{size_col}_min"] = float(values.min())
                        role_profile[f"{size_col}_max"] = float(values.max())
                    break
            total_col = "account_creations" if "account_creations" in frame.columns else None
            if total_col is None:
                for candidate_col in ["orders", "bookings", "revenue", *self.panel.metric_columns]:
                    if candidate_col in frame.columns and pd.api.types.is_numeric_dtype(
                        frame[candidate_col]
                    ):
                        total_col = candidate_col
                        break
            if total_col:
                totals = subset.groupby(geo_col)[total_col].sum().sort_values(ascending=False)
                role_profile["volume_column"] = total_col
                role_profile["total_volume"] = float(totals.sum())
            label_cols = [
                col
                for col in ["cbsa_title", "market_name", "name", "region"]
                if col in market_rows.columns
            ]
            role_profile["markets"] = [
                {
                    "geo_id": str(row[geo_col]),
                    **{col: row[col] for col in label_cols if pd.notna(row[col])},
                }
                for row in market_rows.sort_values(geo_col).head(40).to_dict("records")
            ]
            profile[role] = role_profile
        return profile

    def _balance_diagnostics(
        self,
        spec: ExperimentSpec,
        treatment: list[str],
        controls: list[str],
        start: date,
    ) -> dict[str, Any]:
        frame = self.panel.df
        pre = frame[frame[self.panel.time_col] < pd.Timestamp(start)].copy()
        if pre.empty:
            pre = frame.copy()
        first_metric = next(iter(spec.metrics.values()))
        column = getattr(first_metric, "column", None) or getattr(first_metric, "numerator", None)
        if column not in pre.columns:
            return {"ok": False, "warnings": ["no_balance_metric_column"]}

        geo_col = self.panel.geo_col
        time_col = self.panel.time_col
        work = pre[pre[geo_col].isin([*treatment, *controls])].copy()
        work[time_col] = pd.to_datetime(work[time_col])
        daily = (
            work.groupby([geo_col, time_col], observed=True)[str(column)]
            .sum()
            .reset_index()
            .sort_values([geo_col, time_col])
        )
        if daily.empty:
            return {"ok": False, "warnings": ["no_pre_period_balance_rows"]}

        totals = daily.groupby(geo_col, observed=True)[str(column)].mean()
        t_values = totals.reindex(treatment).dropna().to_numpy(dtype=float)
        c_values = totals.reindex(controls).dropna().to_numpy(dtype=float)
        smd = _standardized_mean_difference(t_values, c_values)
        variance_ratio = _variance_ratio(t_values, c_values)

        trends: dict[str, float] = {}
        for geo, group in daily.groupby(geo_col, observed=True):
            if group[time_col].nunique() < 2:
                continue
            x = (group[time_col] - group[time_col].min()).dt.days.to_numpy(dtype=float)
            y = group[str(column)].to_numpy(dtype=float)
            if np.std(x) > 0:
                trends[str(geo)] = float(np.polyfit(x, y, deg=1)[0])
        t_trends = np.asarray([trends[g] for g in treatment if g in trends], dtype=float)
        c_trends = np.asarray([trends[g] for g in controls if g in trends], dtype=float)
        trend_difference = (
            float(np.mean(t_trends) - np.mean(c_trends))
            if len(t_trends) and len(c_trends)
            else None
        )
        warnings: list[str] = []
        if smd is not None and abs(smd) > 0.25:
            warnings.append("large_standardized_mean_difference")
        if variance_ratio is not None and (variance_ratio < 0.5 or variance_ratio > 2.0):
            warnings.append("large_variance_ratio")
        return {
            "ok": not warnings,
            "metric": str(column),
            "standardized_mean_difference": smd,
            "variance_ratio": variance_ratio,
            "pre_period_trend_difference": trend_difference,
            "treatment_mean": float(np.mean(t_values)) if len(t_values) else None,
            "control_mean": float(np.mean(c_values)) if len(c_values) else None,
            "treatment_market_count": len(treatment),
            "control_market_count": len(controls),
            "warnings": warnings,
        }


def _standardized_mean_difference(
    treatment: np.ndarray,
    control: np.ndarray,
) -> float | None:
    if len(treatment) == 0 or len(control) == 0:
        return None
    t_var = float(np.var(treatment, ddof=1)) if len(treatment) > 1 else 0.0
    c_var = float(np.var(control, ddof=1)) if len(control) > 1 else 0.0
    pooled = np.sqrt((t_var + c_var) / 2.0)
    if not np.isfinite(pooled) or pooled <= 0:
        diff = float(np.mean(treatment) - np.mean(control))
        if np.isclose(diff, 0.0):
            return 0.0
        return float(np.copysign(np.inf, diff))
    return float((np.mean(treatment) - np.mean(control)) / pooled)


def _variance_ratio(treatment: np.ndarray, control: np.ndarray) -> float | None:
    if len(treatment) < 2 or len(control) < 2:
        return None
    control_var = float(np.var(control, ddof=1))
    treatment_var = float(np.var(treatment, ddof=1))
    if not np.isfinite(control_var) or control_var <= 0:
        return None
    return float(treatment_var / control_var)
