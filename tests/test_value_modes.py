from __future__ import annotations

from copy import deepcopy

from src.model import run_model
from src.value_modes import apply_value_mode


def test_nominal_mode_is_identity(base_inputs):
    df = run_model(base_inputs)
    out = apply_value_mode(df, base_inputs, "nominal")
    assert float(out["Total Revenue"].sum()) == float(df["Total Revenue"].sum())


def test_real_modes_reduce_later_period_monetary_values(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["horizon_months"] = 24
    inputs["monthly_cost_inflation"] = 0.01
    inputs["discount_rate_annual_nominal"] = 0.12
    df = run_model(inputs)

    infl = apply_value_mode(df, inputs, "real_inflation")
    pv = apply_value_mode(df, inputs, "real_pv")
    last = len(df) - 1

    assert infl.loc[last, "Total Revenue"] <= df.loc[last, "Total Revenue"]
    assert pv.loc[last, "Total Revenue"] <= df.loc[last, "Total Revenue"]

