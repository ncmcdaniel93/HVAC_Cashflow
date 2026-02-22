from __future__ import annotations

from copy import deepcopy

import pandas as pd

from src.defaults import DEFAULTS
from src.metrics import compute_metrics
from src.model import run_model
from src.pdf_export import build_analyst_pdf_report_bytes, build_pdf_chart_images, build_report_sections
from src.schema import migrate_assumptions
from src.value_modes import apply_value_mode


def _attach_metric_attrs(df: pd.DataFrame, assumptions: dict) -> None:
    df.attrs["attach_rate"] = assumptions.get("attach_rate", 0.0)
    df.attrs["ar_days"] = assumptions.get("ar_days", 0.0)
    df.attrs["ap_days"] = assumptions.get("ap_days", 0.0)
    df.attrs["inventory_days"] = assumptions.get("inventory_days", 0.0)
    df.attrs["tech_wage_per_hour"] = assumptions.get("tech_wage_per_hour", 0.0)


def _build_annual_kpis(df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    annual = (
        metrics["revenue_by_year"]
        .merge(metrics["ebitda_by_year"], on="Year")
        .merge(metrics["fcf_by_year"], on="Year")
    )
    annual["EBITDA Margin %"] = 100.0 * annual["EBITDA"] / annual["Total Revenue"].replace(0, pd.NA)
    annual["DSCR"] = metrics["dscr_by_year"].reset_index(drop=True)
    return annual


def _sample_report_input(*, include_sensitivity: bool = True, include_custom: bool = True) -> dict:
    assumptions, _, _ = migrate_assumptions(deepcopy(DEFAULTS))
    nominal_df = run_model(assumptions)
    _attach_metric_attrs(nominal_df, assumptions)
    view_df = apply_value_mode(nominal_df, assumptions, assumptions["value_mode"])
    _attach_metric_attrs(view_df, assumptions)
    range_df = view_df.iloc[:24].copy()

    metrics_full = compute_metrics(view_df, int(assumptions["horizon_months"]))
    metrics_range = compute_metrics(range_df, len(range_df))
    annual_range = _build_annual_kpis(range_df, metrics_range)
    annual_full = _build_annual_kpis(view_df, metrics_full)

    sensitivity_df = None
    if include_sensitivity:
        sensitivity_df = pd.DataFrame(
            [
                {"Driver": "avg_service_ticket", "Case": "Low", "Delta Year N EBITDA": -5000.0},
                {"Driver": "avg_service_ticket", "Case": "High", "Delta Year N EBITDA": 7000.0},
                {"Driver": "tech_wage_per_hour", "Case": "Low", "Delta Year N EBITDA": 3500.0},
                {"Driver": "tech_wage_per_hour", "Case": "High", "Delta Year N EBITDA": -4200.0},
            ]
        )

    custom_cfgs = []
    if include_custom:
        custom_cfgs = [
            {
                "enabled": True,
                "chart_type": "line",
                "columns": ["Total Revenue", "EBITDA"],
                "title": "Custom Revenue vs EBITDA",
            }
        ]

    input_ts_df = pd.DataFrame(
        {
            "Date": view_df["Date"],
            "Year_Month_Label": view_df["Year_Month_Label"],
            "calls_per_tech_per_day": [assumptions["calls_per_tech_per_day"]] * len(view_df),
            "avg_service_ticket": [assumptions["avg_service_ticket"]] * len(view_df),
            "tech_staffing_events_hires_input": [0.0, 0.0, 2.0] + [0.0] * (len(view_df) - 3),
            "tech_staffing_events_attrition_input": [0.0] * len(view_df),
            "sales_staffing_events_hires_input": [0.0, 1.0] + [0.0] * (len(view_df) - 2),
            "sales_staffing_events_attrition_input": [0.0] * len(view_df),
            "res_new_build_install_schedule_installs_input": [0.0, 0.0, 0.0, 3.0] + [0.0] * (len(view_df) - 4),
            "lc_new_build_install_schedule_installs_input": [0.0] * len(view_df),
        }
    )

    return {
        "scenario_name": "unit_test_scenario",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "value_mode": assumptions["value_mode"],
        "range_start_label": str(range_df["Year_Month_Label"].iloc[0]),
        "range_end_label": str(range_df["Year_Month_Label"].iloc[-1]),
        "assumptions": assumptions,
        "metrics_range": metrics_range,
        "metrics_full": metrics_full,
        "annual_kpis_range": annual_range,
        "annual_kpis_full": annual_full,
        "range_df": range_df,
        "view_df": view_df,
        "nominal_df": nominal_df,
        "input_ts_df": input_ts_df,
        "input_warnings": [],
        "integrity_findings": [],
        "sensitivity_df": sensitivity_df,
        "sensitivity_target": "Year N EBITDA",
        "sensitivity_target_year": 5,
        "transformation_logic": {
            "pipeline_steps": ["step 1", "step 2"],
            "core_identities": [{"name": "EBITDA", "formula": "Gross Profit - Total OPEX"}],
            "value_mode_logic": [{"mode": "nominal", "formula": "no transform"}],
            "implementation_references": [{"file": "src/model.py", "function": "run_model"}],
        },
        "custom_chart_configs": custom_cfgs,
        "custom_chart_data_df": range_df.copy(),
    }


def test_build_analyst_pdf_report_bytes_returns_nonempty_bytes():
    report_input = _sample_report_input()
    pdf_bytes = build_analyst_pdf_report_bytes(report_input, {})
    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert len(pdf_bytes) > 500


def test_build_analyst_pdf_report_bytes_has_pdf_magic_header():
    report_input = _sample_report_input()
    pdf_bytes = build_analyst_pdf_report_bytes(report_input, {})
    assert bytes(pdf_bytes).startswith(b"%PDF")


def test_comprehensive_chart_pack_returns_expected_chart_count():
    report_input = _sample_report_input(include_sensitivity=True, include_custom=True)
    charts = build_pdf_chart_images(
        report_input,
        {
            "chart_pack": "comprehensive",
            "include_enabled_custom_charts": True,
        },
    )
    assert len(charts) == 23  # 22 core charts + 1 enabled custom chart
    assert sum(1 for c in charts if c.get("core_chart")) == 22


def test_chart_render_failure_still_returns_valid_pdf(monkeypatch):
    import src.pdf_export as pdf_export

    report_input = _sample_report_input()
    captured_events: list[dict] = []

    def _log_event(**kwargs):
        captured_events.append(kwargs)

    monkeypatch.setattr(pdf_export, "_ensure_kaleido_ready", lambda options: True)

    def _raise_render(*args, **kwargs):
        raise RuntimeError("forced image failure")

    monkeypatch.setattr(pdf_export, "render_plotly_figure_png", _raise_render)
    charts = build_pdf_chart_images(report_input, {"log_event": _log_event})
    assert any(evt.get("event") == "pdf_chart_render_failed" for evt in captured_events)
    assert all(c.get("image_bytes") is None for c in charts)

    pdf_bytes = build_analyst_pdf_report_bytes(report_input, {"chart_images_override": charts, "log_event": _log_event})
    assert pdf_bytes.startswith(b"%PDF")


def test_missing_sensitivity_data_still_produces_pdf_with_risk_fallback():
    report_input = _sample_report_input(include_sensitivity=False)
    charts = build_pdf_chart_images(report_input, {})
    risk_charts = [c for c in charts if c.get("section") == "Risk & Sensitivity"]
    assert len(risk_charts) == 2
    assert all(c.get("image_bytes") is None for c in risk_charts)

    pdf_bytes = build_analyst_pdf_report_bytes(report_input, {"chart_images_override": charts})
    assert pdf_bytes.startswith(b"%PDF")


def test_wide_and_long_tables_paginate_without_exception():
    report_input = _sample_report_input()
    long_rows = len(report_input["view_df"])
    wide_df = pd.DataFrame({"Date": report_input["view_df"]["Date"], "Year_Month_Label": report_input["view_df"]["Year_Month_Label"]})
    for i in range(1, 28):
        wide_df[f"input_col_{i:02d}"] = i
    report_input["input_ts_df"] = wide_df.iloc[:long_rows].copy()

    expanded_view = report_input["view_df"].copy()
    for i in range(1, 14):
        expanded_view[f"extra_metric_{i:02d}"] = i * 100.0
    report_input["view_df"] = expanded_view

    pdf_bytes = build_analyst_pdf_report_bytes(report_input, {})
    assert pdf_bytes.startswith(b"%PDF")


def test_input_timeseries_is_exported_as_overview_and_event_schedule_tables():
    report_input = _sample_report_input()
    sections = build_report_sections(report_input, chart_images=[], options={})
    appendix = [s for s in sections if s.get("id") == "appendix"][0]
    table_titles = [str(t.get("title", "")) for t in appendix.get("tables", [])]
    assert "Input Time-Series Overview (Start vs End)" in table_titles
    assert "Staffing and Install Event Schedule (Non-zero Months)" in table_titles
    assert "Input Time-Series (Full)" not in table_titles
