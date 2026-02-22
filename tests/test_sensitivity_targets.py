from __future__ import annotations

from copy import deepcopy

from src.sensitivity import TARGET_OPTIONS, run_one_way_sensitivity


def test_sensitivity_target_options_include_break_even_metrics():
    assert "Break-even Revenue" in TARGET_OPTIONS
    assert "Break-even Labor Rate" in TARGET_OPTIONS
    assert "Break-even Wage Rate" in TARGET_OPTIONS


def test_run_one_way_sensitivity_emits_break_even_target_columns(base_inputs):
    inputs = deepcopy(base_inputs)
    sens_df, _ = run_one_way_sensitivity(inputs, delta_pct=0.1, drivers=["tech_wage_per_hour"])

    assert len(sens_df) == 2
    for col in [
        "Break-even Revenue",
        "Delta Break-even Revenue",
        "Break-even Labor Rate",
        "Delta Break-even Labor Rate",
        "Break-even Wage Rate",
        "Delta Break-even Wage Rate",
    ]:
        assert col in sens_df.columns
