from __future__ import annotations

from src.goal_seek import solve_bounded_scalar


def test_goal_seek_converges_on_simple_monotonic_function():
    result = solve_bounded_scalar(lambda x: 2 * x + 3, target=23, lower_bound=0, upper_bound=20, tol=1e-6)
    assert result.status == "solved"
    assert result.value is not None
    assert abs(result.value - 10.0) < 1e-4


def test_goal_seek_fails_when_target_not_bracketed():
    result = solve_bounded_scalar(lambda x: x * x + 1, target=0, lower_bound=0, upper_bound=5)
    assert result.status == "failed"
    assert "not bracketed" in result.message.lower()

