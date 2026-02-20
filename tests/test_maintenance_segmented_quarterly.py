from __future__ import annotations

from copy import deepcopy

from src.model import run_model


def test_lc_quarterly_maintenance_calendar_quarter_boundary(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["start_year"] = 2026
    inputs["start_month"] = 2  # mid-quarter start
    inputs["horizon_months"] = 6
    inputs["enable_maintenance"] = True
    inputs["lc_agreements_start"] = 100
    inputs["lc_new_agreements_per_month"] = 0
    inputs["lc_churn_annual_pct"] = 0
    inputs["lc_maint_quarterly_fee"] = 200
    inputs["lc_maint_visits_per_agreement_per_year"] = 4
    inputs["lc_cost_per_maint_visit"] = 0

    df = run_model(inputs)
    feb = df.loc[df["Year_Month_Label"] == "2026-02", "LC Maintenance Revenue"].iloc[0]
    mar = df.loc[df["Year_Month_Label"] == "2026-03", "LC Maintenance Revenue"].iloc[0]
    apr = df.loc[df["Year_Month_Label"] == "2026-04", "LC Maintenance Revenue"].iloc[0]
    may = df.loc[df["Year_Month_Label"] == "2026-05", "LC Maintenance Revenue"].iloc[0]
    jul = df.loc[df["Year_Month_Label"] == "2026-07", "LC Maintenance Revenue"].iloc[0]

    assert feb == 0
    assert mar == 0
    assert apr > 0
    assert may == 0
    assert jul > 0

