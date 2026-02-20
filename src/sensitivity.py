"""One-way sensitivity analysis helpers."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from src.model import run_model


DEFAULT_SENSITIVITY_DRIVERS = [
    "avg_service_ticket",
    "calls_per_tech_per_day",
    "repl_close_rate",
    "avg_repl_ticket",
    "repl_equipment_pct",
    "tech_wage_per_hour",
    "sales_wage_per_hour",
    "cost_per_lead",
    "ar_days",
    "res_service_upsell_conversion_pct",
    "res_maint_upsell_conversion_pct",
    "res_new_build_avg_ticket",
]


TARGET_OPTIONS = [
    "Year 1 EBITDA",
    "Year 1 Free Cash Flow",
    "Year N EBITDA",
    "Year N Free Cash Flow",
    "Minimum Ending Cash",
    "Total Revenue",
    "Total Disbursements",
]


def available_sensitivity_drivers(inputs: dict) -> list[str]:
    drivers = []
    blocked = {
        "start_year",
        "start_month",
        "horizon_months",
        "peak_month",
        "raise_effective_month",
        "manager_start_year",
        "manager_start_month",
        "ops_manager_start_year",
        "ops_manager_start_month",
        "marketing_manager_start_year",
        "marketing_manager_start_month",
        "loan_term_months",
        "max_techs",
    }
    blocked_prefixes = ("enable_",)
    for k, v in inputs.items():
        if k in blocked:
            continue
        if any(k.startswith(prefix) for prefix in blocked_prefixes):
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (list, dict)):
            continue
        if isinstance(v, (int, float)):
            drivers.append(k)
    return sorted(drivers)


def _year_value(df: pd.DataFrame, year: int, col: str) -> float:
    return float(df.loc[df["Year"] == year, col].sum())


def evaluate_outputs(df: pd.DataFrame, target_year: int) -> dict:
    return {
        "Year 1 EBITDA": _year_value(df, 1, "EBITDA"),
        "Year 1 Free Cash Flow": _year_value(df, 1, "Free Cash Flow"),
        "Year N EBITDA": _year_value(df, target_year, "EBITDA"),
        "Year N Free Cash Flow": _year_value(df, target_year, "Free Cash Flow"),
        "Minimum Ending Cash": float(df["End Cash"].min()),
        "Total Revenue": float(df["Total Revenue"].sum()),
        "Total Disbursements": float(df["Total Disbursements"].sum()) if "Total Disbursements" in df.columns else 0.0,
    }


def run_one_way_sensitivity(base_inputs: dict, delta_pct: float, drivers: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    base_df = run_model(base_inputs)
    full_years = max(int(base_inputs["horizon_months"]) // 12, 1)
    target_year = 5 if int(base_inputs["horizon_months"]) >= 60 else full_years
    base = evaluate_outputs(base_df, target_year)

    if drivers is None or len(drivers) == 0:
        drivers = [d for d in DEFAULT_SENSITIVITY_DRIVERS if d in base_inputs]

    rows = []
    for driver in drivers:
        if driver not in base_inputs or not isinstance(base_inputs[driver], (int, float)):
            continue
        for case, mult in [("Low", 1 - delta_pct), ("High", 1 + delta_pct)]:
            scenario = deepcopy(base_inputs)
            scenario[driver] = float(scenario[driver]) * mult
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
