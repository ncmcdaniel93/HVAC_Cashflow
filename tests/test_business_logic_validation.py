from __future__ import annotations

from copy import deepcopy

from src.model import run_model


def test_service_ticket_lift_increases_service_and_total_revenue(base_inputs):
    low = deepcopy(base_inputs)
    high = deepcopy(base_inputs)
    high["avg_service_ticket"] = float(high["avg_service_ticket"]) * 1.15

    low_df = run_model(low)
    high_df = run_model(high)

    assert float(high_df["Service Revenue"].sum()) > float(low_df["Service Revenue"].sum())
    assert float(high_df["Total Revenue"].sum()) > float(low_df["Total Revenue"].sum())


def test_higher_wages_reduce_total_ebitda(base_inputs):
    low = deepcopy(base_inputs)
    high = deepcopy(base_inputs)
    high["tech_wage_per_hour"] = float(high["tech_wage_per_hour"]) * 1.30
    high["sales_wage_per_hour"] = float(high["sales_wage_per_hour"]) * 1.30

    low_df = run_model(low)
    high_df = run_model(high)

    assert float(high_df["EBITDA"].sum()) < float(low_df["EBITDA"].sum())


def test_disabling_maintenance_zeroes_maintenance_lines(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["enable_maintenance"] = False
    df = run_model(inputs)

    zero_line_items = [
        "Maintenance Revenue",
        "Res Maintenance Revenue",
        "LC Maintenance Revenue",
        "Maintenance Direct Cost",
        "Res Maintenance Agreements",
        "LC Maintenance Agreements",
        "Res Maintenance Visits",
        "LC Maintenance Visits",
    ]
    for col in zero_line_items:
        assert float(df[col].abs().sum()) == 0.0
