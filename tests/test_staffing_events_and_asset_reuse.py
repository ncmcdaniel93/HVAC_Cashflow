from __future__ import annotations

from copy import deepcopy

from src.model import run_model


def test_hire_attrition_reuse_reduces_capex(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["horizon_months"] = 3
    inputs["starting_techs"] = 1
    inputs["max_techs"] = 5
    inputs["asset_reuse_lag_months"] = 3
    inputs["tech_staffing_events"] = [
        {"month": "2026-01", "hires": 0, "attrition": 1},
        {"month": "2026-02", "hires": 1, "attrition": 0},
    ]
    df = run_model(inputs)
    jan = df.loc[df["Year_Month_Label"] == "2026-01"].iloc[0]
    feb = df.loc[df["Year_Month_Label"] == "2026-02"].iloc[0]
    assert jan["Techs"] == 0
    assert feb["Techs"] == 1
    assert feb["Reused Tool Sets"] >= 1
    assert feb["Tools Capex"] == 0
    assert jan["Retained Trucks"] > 0


def test_salvage_mode_generates_proceeds_after_expiry(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["horizon_months"] = 5
    inputs["starting_techs"] = 1
    inputs["max_techs"] = 5
    inputs["asset_reuse_lag_months"] = 1
    inputs["asset_expiry_mode"] = "salvage"
    inputs["asset_salvage_pct"] = 0.5
    inputs["tech_staffing_events"] = [{"month": "2026-01", "hires": 0, "attrition": 1}]
    df = run_model(inputs)
    assert float(df["Asset Salvage Proceeds"].sum()) > 0


def test_same_month_hires_and_attrition_are_tracked_gross_not_net(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["horizon_months"] = 1
    inputs["starting_techs"] = 10
    inputs["max_techs"] = 30
    inputs["tech_staffing_events"] = [{"month": "2026-01", "hires": 2, "attrition": 1}]
    df = run_model(inputs)
    row = df.iloc[0]

    assert row["Techs"] == 11
    assert row["New Tech Hires"] == 2
    assert row["Tech Attrition"] == 1
