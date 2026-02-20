"""Bounded scalar goal-seek helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class GoalSeekResult:
    status: str
    value: float | None
    achieved: float | None
    iterations: int
    message: str


def solve_bounded_scalar(
    evaluator: Callable[[float], float],
    target: float,
    lower_bound: float,
    upper_bound: float,
    tol: float = 1e-3,
    max_iter: int = 60,
) -> GoalSeekResult:
    """Solve evaluator(x)=target for x within [lower_bound, upper_bound] via bisection."""
    lo = float(lower_bound)
    hi = float(upper_bound)
    if hi <= lo:
        return GoalSeekResult("failed", None, None, 0, "Upper bound must be greater than lower bound.")

    try:
        y_lo = float(evaluator(lo))
        y_hi = float(evaluator(hi))
    except Exception as exc:  # pragma: no cover - defensive path
        return GoalSeekResult("failed", None, None, 0, f"Evaluator failed at bounds: {exc}")

    f_lo = y_lo - target
    f_hi = y_hi - target
    if f_lo == 0:
        return GoalSeekResult("solved", lo, y_lo, 0, "Solved at lower bound.")
    if f_hi == 0:
        return GoalSeekResult("solved", hi, y_hi, 0, "Solved at upper bound.")
    if f_lo * f_hi > 0:
        return GoalSeekResult(
            "failed",
            None,
            None,
            0,
            "Target is not bracketed in the selected bounds. Adjust min/max bounds.",
        )

    for i in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        y_mid = float(evaluator(mid))
        f_mid = y_mid - target
        if abs(f_mid) <= tol:
            return GoalSeekResult("solved", mid, y_mid, i, "Converged.")
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    mid = 0.5 * (lo + hi)
    y_mid = float(evaluator(mid))
    return GoalSeekResult(
        "failed",
        mid,
        y_mid,
        max_iter,
        "Reached max iterations before tolerance was met.",
    )

