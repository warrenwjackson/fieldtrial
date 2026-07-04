"""Reusable assignment policies for geo designs and randomization inference."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass, field
from math import comb
from typing import Any

import numpy as np
import pandas as pd

from fieldtrial.methods import _jsonable


@dataclass(frozen=True)
class FeasibleAssignment:
    """One feasible treatment/control assignment from an assignment policy."""

    treatment_markets: tuple[str, ...]
    control_markets: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class AssignmentPolicy:
    """Explicit assignment mechanism shared by design, power, and inference."""

    markets: tuple[str, ...]
    treatment_count: int
    kind: str = "fixed_treatment_count"
    strata: dict[str, str] = field(default_factory=dict)
    pairs: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    required_treatment_markets: tuple[str, ...] = field(default_factory=tuple)
    forbidden_treatment_markets: tuple[str, ...] = field(default_factory=tuple)
    fixed_control_markets: tuple[str, ...] = field(default_factory=tuple)
    shared_control_markets: tuple[str, ...] = field(default_factory=tuple)
    seed: int | None = 0

    def __post_init__(self) -> None:
        markets = tuple(dict.fromkeys(str(market) for market in self.markets))
        if not markets:
            raise ValueError("assignment policy requires at least one market")
        object.__setattr__(self, "markets", markets)
        for attr in (
            "required_treatment_markets",
            "forbidden_treatment_markets",
            "fixed_control_markets",
            "shared_control_markets",
        ):
            values = tuple(dict.fromkeys(str(item) for item in getattr(self, attr)))
            unknown = sorted(set(values).difference(markets))
            if unknown:
                raise ValueError(f"{attr} contains markets outside policy universe: {unknown}")
            object.__setattr__(self, attr, values)
        if self.treatment_count < len(self.required_treatment_markets):
            raise ValueError("treatment_count is smaller than required_treatment_markets")
        unavailable = set(self.forbidden_treatment_markets).union(self.fixed_control_markets)
        feasible = [market for market in markets if market not in unavailable]
        if self.treatment_count > len(feasible):
            raise ValueError("treatment_count exceeds markets eligible for treatment")
        if set(self.required_treatment_markets).intersection(unavailable):
            raise ValueError("required treatment markets cannot be forbidden or fixed controls")
        pair_markets = [market for pair in self.pairs for market in pair]
        unknown_pair_markets = sorted(set(pair_markets).difference(markets))
        if unknown_pair_markets:
            raise ValueError(
                f"pairs contain markets outside policy universe: {unknown_pair_markets}"
            )
        if len(pair_markets) != len(set(pair_markets)):
            raise ValueError("matched-pair markets must not appear in more than one pair")

    @classmethod
    def from_spec(
        cls,
        spec: Any,
        markets: list[str] | tuple[str, ...],
        *,
        strata_values: dict[str, str] | None = None,
        pairs: list[tuple[str, str]] | None = None,
    ) -> AssignmentPolicy:
        treatment_count = getattr(spec, "treatment_count", None) or max(
            1,
            int(round(len(markets) / 2)),
        )
        return cls(
            markets=tuple(markets),
            treatment_count=int(treatment_count),
            kind=str(getattr(spec, "kind", "fixed_treatment_count")),
            strata=strata_values or {},
            pairs=tuple((str(a), str(b)) for a, b in (pairs or [])),
            required_treatment_markets=tuple(getattr(spec, "required_treatment_markets", ()) or ()),
            forbidden_treatment_markets=tuple(
                getattr(spec, "forbidden_treatment_markets", ()) or ()
            ),
            fixed_control_markets=tuple(getattr(spec, "fixed_control_markets", ()) or ()),
            shared_control_markets=tuple(getattr(spec, "shared_control_markets", ()) or ()),
            seed=getattr(spec, "seed", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def enumerate(self, *, max_assignments: int = 10000) -> list[FeasibleAssignment]:
        """Enumerate feasible assignments when the policy space is small."""

        if max_assignments < 1:
            raise ValueError("max_assignments must be positive")
        if self.kind == "matched_pairs" and self.pairs:
            return self._enumerate_pairs(max_assignments=max_assignments)
        if self.kind == "stratified" and self.strata:
            return self._enumerate_stratified(max_assignments=max_assignments)
        return self._enumerate_fixed_count(max_assignments=max_assignments)

    @property
    def n_feasible_assignments(self) -> int:
        if self.kind == "matched_pairs" and self.pairs:
            return self._n_feasible_pair_assignments()
        if self.kind == "stratified" and self.strata:
            by_stratum: dict[str, list[str]] = {}
            for market in self.markets:
                by_stratum.setdefault(self.strata.get(market, "unstratified"), []).append(market)
            counts = self._stratum_treatment_counts(by_stratum)
            total = 1
            unavailable = set(self.forbidden_treatment_markets).union(self.fixed_control_markets)
            for stratum, markets in by_stratum.items():
                required = [
                    market for market in markets if market in self.required_treatment_markets
                ]
                eligible = [
                    market
                    for market in markets
                    if market not in unavailable and market not in required
                ]
                choose = max(counts[stratum] - len(required), 0)
                total *= comb(len(eligible), choose)
            return int(total)
        required = set(self.required_treatment_markets)
        unavailable = set(self.forbidden_treatment_markets).union(self.fixed_control_markets)
        optional = [
            market
            for market in self.markets
            if market not in required and market not in unavailable
        ]
        return int(comb(len(optional), self.treatment_count - len(required)))

    def _n_feasible_pair_assignments(self) -> int:
        """Count matched-pair assignments that satisfy the policy constraints."""

        if self.treatment_count != len(self.pairs):
            return 0
        required = set(self.required_treatment_markets)
        unavailable = set(self.forbidden_treatment_markets).union(self.fixed_control_markets)
        paired = {market for pair in self.pairs for market in pair}
        if required.difference(paired):
            return 0
        total = 1
        for pair in self.pairs:
            required_in_pair = [market for market in pair if market in required]
            if len(required_in_pair) > 1:
                return 0
            candidates = required_in_pair or list(pair)
            options = [market for market in candidates if market not in unavailable]
            total *= len(options)
            if total == 0:
                return 0
        return int(total)

    def enumerate_assignments(self) -> list[dict[str, str]]:
        """Return feasible assignments as unit-role mappings for inference engines."""

        return [
            self._role_mapping(assignment)
            for assignment in self.enumerate(max_assignments=self.n_feasible_assignments)
        ]

    def sample_assignments(
        self,
        n_draws: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> list[dict[str, str]]:
        seed = None if rng is None else int(rng.integers(0, np.iinfo(np.int32).max))
        return [self._role_mapping(assignment) for assignment in self.sample(n_draws, seed=seed)]

    def sample(self, n: int, *, seed: int | None = None) -> list[FeasibleAssignment]:
        """Sample feasible assignments deterministically for large policy spaces."""

        if n < 1:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(self._seed(seed))
        assignments: list[FeasibleAssignment] = []
        seen: set[tuple[str, ...]] = set()
        attempts = 0
        while len(assignments) < n and attempts < n * 100:
            attempts += 1
            if self.kind == "matched_pairs" and self.pairs:
                assignment = self._sample_pairs(rng)
            elif self.kind == "stratified" and self.strata:
                assignment = self._sample_stratified(rng)
            else:
                assignment = self._sample_fixed_count(rng)
            key = assignment.treatment_markets
            if key in seen:
                continue
            seen.add(key)
            assignments.append(assignment)
        if len(assignments) < n:
            assignments.extend(self.enumerate(max_assignments=n)[len(assignments) : n])
        return assignments[:n]

    def score_balance(self, features: pd.DataFrame | dict[str, dict[str, float]]) -> dict[str, Any]:
        """Return balance diagnostics for each feasible assignment sample."""

        frame = _features_frame(features)
        numeric = [
            column
            for column in frame.columns
            if column != "geo_id" and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not numeric:
            return {"ok": False, "warnings": ["no_numeric_features"]}
        assignments = self.enumerate(max_assignments=200)
        rows: list[dict[str, Any]] = []
        for assignment in assignments:
            t = frame[frame["geo_id"].isin(assignment.treatment_markets)]
            c = frame[frame["geo_id"].isin(assignment.control_markets)]
            smds = {
                column: _smd(t[column].to_numpy(dtype=float), c[column].to_numpy(dtype=float))
                for column in numeric
            }
            finite = [abs(value) for value in smds.values() if value is not None]
            rows.append(
                {
                    "treatment_markets": assignment.treatment_markets,
                    "max_abs_smd": max(finite) if finite else None,
                    "standardized_mean_differences": smds,
                }
            )
        rows.sort(
            key=lambda item: item["max_abs_smd"] if item["max_abs_smd"] is not None else np.inf
        )
        return {
            "ok": True,
            "assignment_count": len(rows),
            "best_assignment": rows[0] if rows else None,
            "assignments": rows,
        }

    def _enumerate_fixed_count(self, *, max_assignments: int) -> list[FeasibleAssignment]:
        required = set(self.required_treatment_markets)
        unavailable = set(self.forbidden_treatment_markets).union(self.fixed_control_markets)
        optional = [
            market
            for market in self.markets
            if market not in required and market not in unavailable
        ]
        choose = self.treatment_count - len(required)
        assignments: list[FeasibleAssignment] = []
        for combo in itertools.combinations(optional, choose):
            treatment = tuple(sorted([*required, *combo]))
            assignments.append(self._assignment(treatment, exact=True))
            if len(assignments) >= max_assignments:
                break
        return assignments

    def _enumerate_pairs(self, *, max_assignments: int) -> list[FeasibleAssignment]:
        assignments: list[FeasibleAssignment] = []
        for orientations in itertools.product([0, 1], repeat=len(self.pairs)):
            treatment = []
            for pair, orientation in zip(self.pairs, orientations, strict=True):
                treatment.append(pair[orientation])
            treatment_tuple = tuple(sorted(treatment))
            if self._is_feasible_treatment(treatment_tuple):
                assignments.append(self._assignment(treatment_tuple, exact=True))
            if len(assignments) >= max_assignments:
                break
        return assignments

    def _enumerate_stratified(self, *, max_assignments: int) -> list[FeasibleAssignment]:
        by_stratum: dict[str, list[str]] = {}
        for market in self.markets:
            by_stratum.setdefault(self.strata.get(market, "unstratified"), []).append(market)
        stratum_counts = self._stratum_treatment_counts(by_stratum)
        products = []
        for stratum, markets in sorted(by_stratum.items()):
            unavailable = set(self.forbidden_treatment_markets).union(self.fixed_control_markets)
            eligible = [market for market in markets if market not in unavailable]
            required = [market for market in markets if market in self.required_treatment_markets]
            choose = max(stratum_counts[stratum] - len(required), 0)
            choices = [
                tuple(sorted([*required, *combo]))
                for combo in itertools.combinations(
                    [market for market in eligible if market not in required],
                    choose,
                )
            ]
            products.append(choices)

        assignments: list[FeasibleAssignment] = []
        for parts in itertools.product(*products):
            treatment = tuple(sorted(itertools.chain.from_iterable(parts)))
            if len(treatment) == self.treatment_count and self._is_feasible_treatment(treatment):
                assignments.append(self._assignment(treatment, exact=True))
            if len(assignments) >= max_assignments:
                break
        return assignments

    def _sample_fixed_count(self, rng: np.random.Generator) -> FeasibleAssignment:
        required = set(self.required_treatment_markets)
        unavailable = set(self.forbidden_treatment_markets).union(self.fixed_control_markets)
        optional = [
            market
            for market in self.markets
            if market not in required and market not in unavailable
        ]
        choose = self.treatment_count - len(required)
        sampled = rng.choice(optional, size=choose, replace=False).tolist() if choose else []
        return self._assignment(tuple(sorted([*required, *sampled])), exact=False)

    def _sample_pairs(self, rng: np.random.Generator) -> FeasibleAssignment:
        for _ in range(100):
            treatment = []
            for first, second in self.pairs:
                treatment.append(first if int(rng.integers(0, 2)) == 0 else second)
            treatment_tuple = tuple(sorted(treatment))
            if self._is_feasible_treatment(treatment_tuple):
                return self._assignment(treatment_tuple, exact=False)
        raise ValueError("could not sample a feasible matched-pair assignment")

    def _sample_stratified(self, rng: np.random.Generator) -> FeasibleAssignment:
        by_stratum: dict[str, list[str]] = {}
        for market in self.markets:
            by_stratum.setdefault(self.strata.get(market, "unstratified"), []).append(market)
        counts = self._stratum_treatment_counts(by_stratum)
        treatment: list[str] = []
        unavailable = set(self.forbidden_treatment_markets).union(self.fixed_control_markets)
        for stratum, markets in by_stratum.items():
            required = [market for market in markets if market in self.required_treatment_markets]
            eligible = [
                market for market in markets if market not in unavailable and market not in required
            ]
            choose = max(counts[stratum] - len(required), 0)
            sampled = rng.choice(eligible, size=choose, replace=False).tolist() if choose else []
            treatment.extend([*required, *sampled])
        return self._assignment(tuple(sorted(treatment)), exact=False)

    def _stratum_treatment_counts(self, by_stratum: dict[str, list[str]]) -> dict[str, int]:
        unavailable = set(self.forbidden_treatment_markets).union(self.fixed_control_markets)
        required = set(self.required_treatment_markets)
        ordered = sorted(by_stratum.items())
        capacity = {
            stratum: sum(1 for market in markets if market not in unavailable)
            for stratum, markets in ordered
        }
        floor = {
            stratum: sum(1 for market in markets if market in required)
            for stratum, markets in ordered
        }
        counts: dict[str, int] = {}
        total = sum(len(markets) for markets in by_stratum.values())
        remaining = self.treatment_count
        for index, (stratum, markets) in enumerate(ordered):
            if index == len(ordered) - 1:
                count = min(remaining, capacity[stratum])
            else:
                count = int(round(self.treatment_count * len(markets) / total))
                count = min(max(count, floor[stratum]), capacity[stratum], remaining)
            counts[stratum] = count
            remaining -= count
        if remaining > 0:
            for stratum, _ in ordered:
                if remaining <= 0:
                    break
                spare = capacity[stratum] - counts[stratum]
                if spare <= 0:
                    continue
                add = min(spare, remaining)
                counts[stratum] += add
                remaining -= add
        if remaining > 0:
            raise ValueError(
                "stratified policy cannot place treatment_count markets "
                "given eligibility constraints"
            )
        return counts

    def _assignment(self, treatment: tuple[str, ...], *, exact: bool) -> FeasibleAssignment:
        if not self._is_feasible_treatment(treatment):
            raise ValueError("infeasible treatment assignment")
        treatment_set = set(treatment)
        controls = tuple(sorted(market for market in self.markets if market not in treatment_set))
        return FeasibleAssignment(
            treatment_markets=tuple(sorted(treatment)),
            control_markets=controls,
            metadata={"policy_kind": self.kind, "exact": exact},
        )

    @staticmethod
    def _role_mapping(assignment: FeasibleAssignment) -> dict[str, str]:
        return {
            **{market: "treatment" for market in assignment.treatment_markets},
            **{market: "control" for market in assignment.control_markets},
        }

    def _is_feasible_treatment(self, treatment: tuple[str, ...]) -> bool:
        treatment_set = set(treatment)
        if len(treatment) != self.treatment_count:
            return False
        if not set(self.required_treatment_markets).issubset(treatment_set):
            return False
        if treatment_set.intersection(self.forbidden_treatment_markets):
            return False
        if treatment_set.intersection(self.fixed_control_markets):
            return False
        return True

    def _seed(self, seed: int | None) -> int | None:
        if seed is not None:
            return seed
        if self.seed is not None:
            digest = hashlib.sha256(repr(self.to_dict()).encode()).hexdigest()
            return int(digest[:8], 16)
        return None


def _features_frame(features: pd.DataFrame | dict[str, dict[str, float]]) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        frame = features.copy()
        if "geo_id" not in frame.columns:
            frame = frame.reset_index().rename(columns={"index": "geo_id"})
        frame["geo_id"] = frame["geo_id"].astype(str)
        return frame
    rows = [{"geo_id": str(geo), **values} for geo, values in features.items()]
    return pd.DataFrame(rows)


def _smd(treatment: np.ndarray, control: np.ndarray) -> float | None:
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
