from __future__ import annotations

from copy import deepcopy

from src.model import run_model


def test_sales_staff_lifts_replacement_close_rate_with_cap(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["horizon_months"] = 1
    inputs["sales_starting_staff"] = 10
    inputs["sales_repl_close_lift_per_fte"] = 0.05
    inputs["repl_close_rate"] = 0.2
    inputs["sales_repl_close_rate_cap"] = 0.4
    df = run_model(inputs)
    row = df.iloc[0]
    implied_rate = row["Replacement Jobs"] / row["Replacement Leads"] if row["Replacement Leads"] else 0.0
    assert abs(implied_rate - 0.4) < 1e-6


def test_upsell_revenue_is_incremental_line_item(base_inputs):
    low = deepcopy(base_inputs)
    high = deepcopy(base_inputs)
    low["res_service_upsell_conversion_pct"] = 0.0
    low["res_maint_upsell_conversion_pct"] = 0.0
    low["lc_service_upsell_conversion_pct"] = 0.0
    low["lc_maint_upsell_conversion_pct"] = 0.0

    high["res_service_upsell_conversion_pct"] = 0.2
    high["res_maint_upsell_conversion_pct"] = 0.2
    high["lc_service_upsell_conversion_pct"] = 0.2
    high["lc_maint_upsell_conversion_pct"] = 0.2

    df_low = run_model(low)
    df_high = run_model(high)
    assert float(df_high["Upsell Revenue"].sum()) > float(df_low["Upsell Revenue"].sum())
    assert float(df_high["Total Revenue"].sum()) > float(df_low["Total Revenue"].sum())

