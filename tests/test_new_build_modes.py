from __future__ import annotations

from copy import deepcopy

from src.model import run_model


def test_new_build_schedule_mode_uses_monthly_schedule(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["horizon_months"] = 3
    inputs["new_build_mode"] = "schedule"
    inputs["res_new_build_install_schedule"] = [{"month": "2026-01", "installs": 5.0}]
    inputs["lc_new_build_install_schedule"] = []
    inputs["res_new_build_avg_ticket"] = 1000.0
    inputs["lc_new_build_avg_ticket"] = 0.0
    df = run_model(inputs)
    jan = df.loc[df["Year_Month_Label"] == "2026-01", "Res New Build Installs"].iloc[0]
    feb = df.loc[df["Year_Month_Label"] == "2026-02", "Res New Build Installs"].iloc[0]
    assert jan == 5.0
    assert feb == 0.0


def test_new_build_annual_total_mode_distributes_over_months(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["horizon_months"] = 12
    inputs["new_build_mode"] = "annual_total"
    inputs["res_new_build_annual_installs"] = 24.0
    inputs["lc_new_build_annual_installs"] = 12.0
    df = run_model(inputs)
    assert abs(float(df["Res New Build Installs"].sum()) - 24.0) < 1e-6
    assert abs(float(df["LC New Build Installs"].sum()) - 12.0) < 1e-6

