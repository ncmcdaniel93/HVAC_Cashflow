"""Nominal and real-dollar presentation transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd


NON_MONETARY_COLUMNS = {
    "Year",
    "Month_Number",
    "Date",
    "Year_Month_Label",
    "Techs",
    "Sales Staff",
    "Trucks",
    "Retained Trucks",
    "Calls",
    "Res Calls",
    "LC Calls",
    "Replacement Leads",
    "Replacement Jobs",
    "Maintenance Agreements",
    "Res Maintenance Agreements",
    "LC Maintenance Agreements",
    "Res New Agreements",
    "LC New Agreements",
    "Res Maintenance Visits",
    "LC Maintenance Visits",
    "Workdays",
    "Tech Hours",
    "Sales Hours",
    "New Tech Hires",
    "Tech Attrition",
    "New Sales Hires",
    "Sales Attrition",
    "Res New Build Installs",
    "LC New Build Installs",
    "Reused Tool Sets",
    "Reused Truck Units",
}


def money_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in NON_MONETARY_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if col.endswith("%") or "Rate" in col:
            continue
        cols.append(col)
    return cols


def apply_value_mode(df: pd.DataFrame, inputs: dict, value_mode: str) -> pd.DataFrame:
    """Return a presentation dataframe transformed for the requested value mode."""
    out = df.copy()
    if value_mode == "nominal":
        return out

    t = np.arange(len(out), dtype=float)
    monthly_infl = float(inputs.get("monthly_cost_inflation", 0.0))
    annual_discount = float(inputs.get("discount_rate_annual_nominal", 0.0))

    if value_mode == "real_inflation":
        factor = (1.0 + monthly_infl) ** t
    elif value_mode == "real_pv":
        # Convert annual nominal discount rate into monthly compounding basis.
        factor = (1.0 + annual_discount) ** (t / 12.0)
    else:
        return out

    cols = money_columns(out)
    for col in cols:
        out[col] = out[col] / factor
    return out


def value_mode_label(value_mode: str) -> str:
    if value_mode == "real_inflation":
        return "Inflation-adjusted real dollars"
    if value_mode == "real_pv":
        return "Discounted present-value real dollars"
    return "Nominal dollars"
