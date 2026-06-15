"""Simulation helpers."""

from __future__ import annotations

from fieldtrial.power.placebo import PowerCurvePoint


def estimate_mde_from_simulations(
    curve: list[PowerCurvePoint],
    *,
    target_power: float = 0.8,
) -> float:
    for point in sorted(curve, key=lambda item: item.lift):
        if point.power >= target_power:
            return point.lift
    return max((point.lift for point in curve), default=float("inf"))


def power_curve(points: list[tuple[float, float]]) -> list[PowerCurvePoint]:
    return [PowerCurvePoint(lift=float(lift), power=float(power)) for lift, power in points]
