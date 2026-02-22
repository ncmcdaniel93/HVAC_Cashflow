from __future__ import annotations

from copy import deepcopy

from src.metrics import compute_metrics
from src.model import run_model


def test_break_even_labor_rate_matches_break_even_revenue_per_avg_tech_hour(base_inputs):
    df = run_model(deepcopy(base_inputs))
    metrics = compute_metrics(df, len(df))

    avg_tech_hours = float(df["Tech Hours"].mean())
    expected = float(metrics["break_even_revenue"]) / avg_tech_hours if avg_tech_hours else 0.0

    assert abs(float(metrics["break_even_labor_rate_per_tech_hour"]) - expected) < 1e-9


def test_break_even_labor_rate_is_zero_when_no_tech_hours(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["starting_techs"] = 0
    inputs["max_techs"] = 0
    inputs["tech_staffing_events"] = []

    df = run_model(inputs)
    metrics = compute_metrics(df, len(df))

    assert float(df["Tech Hours"].sum()) == 0.0
    assert float(metrics["break_even_labor_rate_per_tech_hour"]) == 0.0


def test_break_even_wage_rate_matches_analytical_solution(base_inputs):
    df = run_model(deepcopy(base_inputs))
    metrics = compute_metrics(df, len(df))

    coeff = float(df["Tech Labor Cost per Wage Unit"].sum())
    total_rev = float(df["Total Revenue"].sum())
    total_direct = float(df["Total Direct Costs"].sum())
    direct_labor = float(df["Direct Labor"].sum())
    total_opex = float(df["Total OPEX"].sum())
    expected = (total_rev - (total_direct - direct_labor) - total_opex) / coeff if coeff else 0.0

    assert abs(float(metrics["break_even_wage_rate_per_hour"]) - expected) < 1e-9


def test_break_even_wage_rate_is_zero_when_no_tech_labor_coefficient(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["starting_techs"] = 0
    inputs["max_techs"] = 0
    inputs["tech_staffing_events"] = []

    df = run_model(inputs)
    metrics = compute_metrics(df, len(df))

    assert float(df["Tech Labor Cost per Wage Unit"].sum()) == 0.0
    assert float(metrics["break_even_wage_rate_per_hour"]) == 0.0


def test_break_even_wage_rate_falls_back_when_coefficient_column_missing(base_inputs):
    inputs = deepcopy(base_inputs)
    df = run_model(inputs)
    expected_metrics = compute_metrics(df, len(df))

    stale_df = df.drop(columns=["Tech Labor Cost per Wage Unit"]).copy()
    stale_df.attrs["tech_wage_per_hour"] = float(inputs["tech_wage_per_hour"])
    fallback_metrics = compute_metrics(stale_df, len(stale_df))

    assert abs(float(fallback_metrics["break_even_wage_rate_per_hour"]) - float(expected_metrics["break_even_wage_rate_per_hour"])) < 1e-9
