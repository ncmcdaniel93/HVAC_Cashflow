"""One-way sensitivity analysis helpers."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from src.model import run_model


SENSITIVITY_DRIVERS = [
    "avg_service_ticket",
    "calls_per_tech_per_day",
    "repl_close_rate",
    "avg_repl_ticket",
    "repl_equipment_pct",
    "tech_wage_per_hour",
    "cost_per_lead",
    "ar_days",
]


TARGET_OPTIONS = [
    "Year 1 EBITDA",
    "Year 1 Free Cash Flow",
    "Year N EBITDA",
    "Year N Free Cash Flow",
    "Minimum Ending Cash",
]


def _year_value(df: pd.DataFrame, year: int, col: str) -> float:
    return float(df.loc[df["Year"] == year, col].sum())


def evaluate_outputs(df: pd.DataFrame, target_year: int) -> dict:
    return {
        "Year 1 EBITDA": _year_value(df, 1, "EBITDA"),
        "Year 1 Free Cash Flow": _year_value(df, 1, "Free Cash Flow"),
        "Year N EBITDA": _year_value(df, target_year, "EBITDA"),
        "Year N Free Cash Flow": _year_value(df, target_year, "Free Cash Flow"),
        "Minimum Ending Cash": float(df["End Cash"].min()),
    }


def run_one_way_sensitivity(base_inputs: dict, delta_pct: float) -> pd.DataFrame:
    base_df = run_model(base_inputs)
    full_years = max(base_inputs["horizon_months"] // 12, 1)
    target_year = 5 if base_inputs["horizon_months"] >= 60 else full_years
    base = evaluate_outputs(base_df, target_year)

    rows = []
    for driver in SENSITIVITY_DRIVERS:
        for case, mult in [("Low", 1 - delta_pct), ("High", 1 + delta_pct)]:
            scenario = deepcopy(base_inputs)
            scenario[driver] = scenario[driver] * mult
            if driver.endswith("_pct") or "rate" in driver:
                scenario[driver] = min(max(scenario[driver], 0.0), 1.0)
            out = evaluate_outputs(run_model(scenario), target_year)
            rows.append(
                {
                    "Driver": driver,
                    "Case": case,
                    **{k: out[k] for k in base.keys()},
                    **{f"Delta {k}": out[k] - base[k] for k in base.keys()},
                }
            )

    return pd.DataFrame(rows), target_year
