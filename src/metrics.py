"""Metric calculations for dashboard and summary outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def compute_metrics(df: pd.DataFrame, horizon_months: int) -> dict:
    by_year = df.groupby("Year", as_index=False).sum(numeric_only=True)
    full_years = horizon_months // 12

    revenue_by_year = by_year[["Year", "Total Revenue"]].copy()
    ebitda_by_year = by_year[["Year", "EBITDA"]].copy()
    fcf_by_year = by_year[["Year", "Free Cash Flow"]].copy()

    ebitda_margin = by_year.apply(lambda r: _safe_div(r["EBITDA"], r["Total Revenue"]), axis=1)
    gross_margin = by_year.apply(lambda r: _safe_div(r["Gross Profit"], r["Total Revenue"]), axis=1)

    tech_avg_year = df.groupby("Year")["Techs"].mean().replace(0, np.nan)
    if "Trucks" in df.columns:
        truck_avg_year = df.groupby("Year")["Trucks"].mean().replace(0, np.nan)
    else:
        truck_avg_year = tech_avg_year
    revenue_per_tech = by_year.set_index("Year")["Total Revenue"] / tech_avg_year
    revenue_per_truck = by_year.set_index("Year")["Total Revenue"] / truck_avg_year

    new_customers = df["Calls"] * df.attrs.get("attach_rate", 0.35) + df["Replacement Jobs"]
    cac = _safe_div(df["Marketing Spend"].sum(), new_customers.sum())

    total_rev = df["Total Revenue"].sum()
    total_direct = df["Total Direct Costs"].sum()
    cm_pct = _safe_div(total_rev - total_direct, total_rev)
    fixed_costs = df["Fixed OPEX"].mean() + df["Fleet Cost"].mean() + df["Marketing Spend"].mean()
    break_even = _safe_div(fixed_costs, cm_pct)

    debt_service_year = by_year["Term Loan Payment"] + by_year["LOC Interest"]
    dscr = by_year["EBITDA"] / debt_service_year.replace(0, np.nan)

    min_idx = df["End Cash"].idxmin()
    min_cash = float(df.loc[min_idx, "End Cash"])
    min_cash_month = str(df.loc[min_idx, "Year_Month_Label"])

    metrics = {
        "revenue_by_year": revenue_by_year,
        "ebitda_by_year": ebitda_by_year,
        "fcf_by_year": fcf_by_year,
        "ebitda_margin_by_year": ebitda_margin,
        "gross_margin_by_year": gross_margin,
        "gross_margin_full_period_avg": _safe_div(df["Gross Profit"].sum(), df["Total Revenue"].sum()),
        "revenue_per_tech_by_year": revenue_per_tech,
        "revenue_per_truck_by_year": revenue_per_truck,
        "cac": cac,
        "break_even_revenue": break_even,
        "ccc": df.attrs.get("ar_days", 0) + df.attrs.get("inventory_days", 0) - df.attrs.get("ap_days", 0),
        "dscr_by_year": dscr,
        "minimum_ending_cash": min_cash,
        "minimum_ending_cash_month": min_cash_month,
        "negative_cash_months": int((df["End Cash"] < 0).sum()),
        "full_years": int(full_years),
    }

    if full_years >= 2:
        start_rev = by_year.loc[by_year["Year"] == 1, "Total Revenue"].sum()
        last_rev = by_year.loc[by_year["Year"] == full_years, "Total Revenue"].sum()
        start_ebitda = by_year.loc[by_year["Year"] == 1, "EBITDA"].sum()
        last_ebitda = by_year.loc[by_year["Year"] == full_years, "EBITDA"].sum()
        years = max(full_years - 1, 1)
        metrics["revenue_cagr"] = (last_rev / start_rev) ** (1 / years) - 1 if start_rev > 0 else 0
        metrics["ebitda_cagr"] = (last_ebitda / start_ebitda) ** (1 / years) - 1 if start_ebitda > 0 else 0
    else:
        metrics["revenue_cagr"] = None
        metrics["ebitda_cagr"] = None

    return metrics
