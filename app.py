import json
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitAPIException

from src.defaults import DEFAULTS
from src.goal_seek import solve_bounded_scalar
from src.integrity_checks import run_integrity_checks
from src.input_metadata import INPUT_GUIDANCE, advisory_warnings, help_with_guidance
from src.metrics import compute_metrics
from src.model import run_model
from src.persistence import (
    build_scenario_bundle,
    build_workspace_bundle,
    delete_saved,
    list_saved_names,
    load_saved,
    parse_import_json,
    save_named_bundle,
)
from src.runtime_logging import (
    append_runtime_event,
    install_global_exception_logging,
    read_runtime_events,
    runtime_log_path,
)
from src.schema import (
    SCENARIO_TYPE,
    WORKSPACE_TYPE,
    migrate_assumptions,
    month_name_option,
    parse_month_name_option,
)
from src.sensitivity import (
    DEFAULT_SENSITIVITY_DRIVERS,
    TARGET_OPTIONS,
    available_sensitivity_drivers,
    run_one_way_sensitivity,
)
from src.value_modes import apply_value_mode, value_mode_label


install_global_exception_logging()


UI_DEFAULTS = {
    "preset_choice": "Base",
    "preset_edit_slot": "Base",
    "input_focus_section": "All Sections",
    "expand_all_input_sections": False,
    "last_active_input_section": "Model Controls",
    "input_search_query": "",
    "range_preset": "Full horizon",
    "range_start_label": "",
    "range_end_label": "",
    "table_selected_columns": [],
    "sensitivity_delta": 0.1,
    "sensitivity_drivers": [],
    "sensitivity_auto_refresh": False,
    "sensitivity_result_signature": None,
    "autosave_enabled": False,
    "active_workspace_name": "",
    "goal_seek_result": None,
    "goal_target_metric": "Total EBITDA",
    "goal_adjustable_input": "avg_service_ticket",
    "goal_target_value": 0.0,
    "goal_column_agg": "sum",
    "goal_column_name": "EBITDA",
    "goal_year": 1,
    "helper_revenue_shift_pct": 0.0,
    "helper_cost_shift_pct": 0.0,
    "auto_run_model": True,
    "applied_assumptions_json": "",
    "tech_staffing_events_master": None,
    "sales_staffing_events_master": None,
    "res_new_build_install_schedule_master": None,
    "lc_new_build_install_schedule_master": None,
}


def _build_presets() -> dict:
    base = deepcopy(DEFAULTS)
    upside = deepcopy(DEFAULTS)
    downside = deepcopy(DEFAULTS)

    upside.update(
        {
            "calls_per_tech_per_day": round(DEFAULTS["calls_per_tech_per_day"] * 1.08, 3),
            "avg_service_ticket": round(DEFAULTS["avg_service_ticket"] * 1.06, 2),
            "avg_repl_ticket": round(DEFAULTS["avg_repl_ticket"] * 1.05, 2),
            "repl_close_rate": min(1.0, DEFAULTS["repl_close_rate"] + 0.04),
            "res_service_upsell_conversion_pct": min(1.0, DEFAULTS["res_service_upsell_conversion_pct"] + 0.02),
            "res_maint_upsell_conversion_pct": min(1.0, DEFAULTS["res_maint_upsell_conversion_pct"] + 0.02),
            "cost_per_lead": round(DEFAULTS["cost_per_lead"] * 0.9, 2),
            "res_churn_annual_pct": max(0.0, DEFAULTS["res_churn_annual_pct"] * 0.85),
        }
    )
    downside.update(
        {
            "calls_per_tech_per_day": round(DEFAULTS["calls_per_tech_per_day"] * 0.9, 3),
            "avg_service_ticket": round(DEFAULTS["avg_service_ticket"] * 0.94, 2),
            "avg_repl_ticket": round(DEFAULTS["avg_repl_ticket"] * 0.94, 2),
            "repl_close_rate": max(0.0, DEFAULTS["repl_close_rate"] - 0.05),
            "res_service_upsell_conversion_pct": max(0.0, DEFAULTS["res_service_upsell_conversion_pct"] - 0.03),
            "res_maint_upsell_conversion_pct": max(0.0, DEFAULTS["res_maint_upsell_conversion_pct"] - 0.03),
            "cost_per_lead": round(DEFAULTS["cost_per_lead"] * 1.1, 2),
            "res_churn_annual_pct": min(1.0, DEFAULTS["res_churn_annual_pct"] * 1.2),
        }
    )
    return {"Base": base, "Upside": upside, "Downside": downside}


PRESET_SCENARIOS = _build_presets()

AI_EXPORT_SCOPE_OPTIONS = [
    "Active scenario only",
    "Saved scenarios (select multiple)",
    "Saved workspaces (select multiple)",
    "Active + selected saved items",
]

AI_EXPORT_SOURCE_FILES = [
    "app.py",
    "src/model.py",
    "src/metrics.py",
    "src/value_modes.py",
    "src/schema.py",
    "src/integrity_checks.py",
    "src/sensitivity.py",
    "src/persistence.py",
    "src/defaults.py",
]

SCENARIO_TEMPLATE_COMPLEX_KEYS = [
    "tech_staffing_events",
    "sales_staffing_events",
    "res_new_build_install_schedule",
    "lc_new_build_install_schedule",
]

SCENARIO_TEMPLATE_DETAIL_SPECS = {
    "tech_staffing_events": {
        "sheet": "tech_staffing_events",
        "columns": ["scenario_name", "month", "hires", "attrition"],
    },
    "sales_staffing_events": {
        "sheet": "sales_staffing_events",
        "columns": ["scenario_name", "month", "hires", "attrition"],
    },
    "res_new_build_install_schedule": {
        "sheet": "res_new_build_install_schedule",
        "columns": ["scenario_name", "month", "installs"],
    },
    "lc_new_build_install_schedule": {
        "sheet": "lc_new_build_install_schedule",
        "columns": ["scenario_name", "month", "installs"],
    },
}

SCENARIO_TEMPLATE_ENUM_OPTIONS = {
    "value_mode": ["nominal", "real_inflation", "real_pv"],
    "paid_leads_mode": ["fixed", "per_tech"],
    "capex_trucks_mode": ["payments_only", "purchase_with_downpayment"],
    "new_build_mode": ["schedule", "base_seasonal", "annual_total"],
    "asset_expiry_mode": ["release", "retain", "salvage"],
}

SCENARIO_TEMPLATE_SHEET_SCENARIOS = "scenarios"
SCENARIO_TEMPLATE_SHEET_README = "README"
SCENARIO_TEMPLATE_SHEET_REFERENCE = "field_reference"


INPUT_SECTION_KEY_GROUPS = {
    "Model Controls": {
        "start_year",
        "start_month",
        "horizon_months",
        "monthly_price_growth",
        "monthly_cost_inflation",
        "seasonality_amplitude",
        "peak_month",
        "value_mode",
        "discount_rate_annual_nominal",
    },
    "Staffing and Ops": {
        "starting_techs",
        "max_techs",
        "sales_starting_staff",
        "calls_per_tech_per_day",
        "tech_hours_per_day",
        "sales_hours_per_day",
        "tech_wage_per_hour",
        "sales_wage_per_hour",
        "payroll_burden_pct",
        "tools_per_new_tech_capex",
        "asset_reuse_lag_months",
        "asset_expiry_mode",
        "asset_salvage_pct",
        "res_capacity_pct",
        "tech_staffing_events",
        "sales_staffing_events",
    },
    "Service and Replacement": {
        "avg_service_ticket",
        "service_material_pct",
        "attach_rate",
        "repl_leads_per_tech_per_month",
        "repl_close_rate",
        "sales_repl_close_lift_per_fte",
        "sales_repl_close_rate_cap",
        "avg_repl_ticket",
        "repl_equipment_pct",
        "permit_cost_per_repl_job",
        "disposal_cost_per_repl_job",
        "financing_penetration",
        "financing_fee_pct",
    },
    "Maintenance": {
        "enable_maintenance",
        "maint_visits_capacity_per_tech_per_month",
        "res_agreements_start",
        "res_new_agreements_per_month",
        "res_churn_annual_pct",
        "res_maint_monthly_fee",
        "res_cost_per_maint_visit",
        "res_maint_visits_per_agreement_per_year",
        "res_maint_call_conversion_pct",
        "res_maint_agreements_per_tech_per_month",
        "res_maint_hybrid_weight_calls",
        "lc_agreements_start",
        "lc_new_agreements_per_month",
        "lc_churn_annual_pct",
        "lc_maint_quarterly_fee",
        "lc_cost_per_maint_visit",
        "lc_maint_visits_per_agreement_per_year",
        "lc_maint_call_conversion_pct",
        "lc_maint_agreements_per_tech_per_month",
        "lc_maint_hybrid_weight_calls",
    },
    "Upsell": {
        "res_service_upsell_conversion_pct",
        "res_service_upsell_revenue_per_visit",
        "res_service_upsell_gross_margin_pct",
        "lc_service_upsell_conversion_pct",
        "lc_service_upsell_revenue_per_visit",
        "lc_service_upsell_gross_margin_pct",
        "res_maint_upsell_conversion_pct",
        "res_maint_upsell_revenue_per_visit",
        "res_maint_upsell_gross_margin_pct",
        "lc_maint_upsell_conversion_pct",
        "lc_maint_upsell_revenue_per_visit",
        "lc_maint_upsell_gross_margin_pct",
    },
    "New-Build Installs": {
        "new_build_mode",
        "new_build_seasonality_amplitude",
        "res_new_build_installs_per_month",
        "lc_new_build_installs_per_month",
        "res_new_build_annual_installs",
        "lc_new_build_annual_installs",
        "res_new_build_avg_ticket",
        "lc_new_build_avg_ticket",
        "res_new_build_gross_margin_pct",
        "lc_new_build_gross_margin_pct",
        "res_new_build_install_schedule",
        "lc_new_build_install_schedule",
    },
    "Marketing and Fleet": {
        "paid_leads_mode",
        "paid_leads_per_month",
        "paid_leads_per_tech_per_month",
        "cost_per_lead",
        "branding_fixed_monthly",
        "trucks_per_tech",
        "truck_payment_monthly",
        "fuel_per_truck_monthly",
        "maint_per_truck_monthly",
        "truck_insurance_per_truck_monthly",
        "truck_purchase_price",
        "capex_trucks_mode",
        "truck_downpayment_pct",
        "truck_financed_pct",
    },
    "Overhead and Management": {
        "office_payroll_monthly",
        "rent_monthly",
        "utilities_monthly",
        "insurance_monthly",
        "software_monthly",
        "other_fixed_monthly",
        "manager_salary_monthly",
        "manager_start_year",
        "manager_start_month",
        "ops_manager_salary_monthly",
        "ops_manager_start_year",
        "ops_manager_start_month",
        "marketing_manager_salary_monthly",
        "marketing_manager_start_year",
        "marketing_manager_start_month",
        "raise_effective_month",
        "annual_raise_pct_tech",
        "annual_raise_pct_sales",
        "annual_raise_pct_manager",
        "annual_raise_pct_ops_manager",
        "annual_raise_pct_marketing_manager",
    },
    "Working Capital, Debt, and Distributions": {
        "ar_days",
        "ap_days",
        "inventory_days",
        "opening_ar_balance",
        "opening_ap_balance",
        "opening_inventory_balance",
        "starting_cash",
        "enable_term_loan",
        "loan_principal",
        "loan_annual_rate",
        "loan_term_months",
        "enable_loc",
        "loc_limit",
        "loc_annual_rate",
        "min_cash_target",
        "loc_repay_buffer",
        "enable_distributions",
        "distributions_pct_of_ebitda",
        "distributions_only_if_cash_above",
    },
}


INPUT_KEY_TO_SECTION = {key: section for section, keys in INPUT_SECTION_KEY_GROUPS.items() for key in keys}
INPUT_KEY_TO_SECTION.update(
    {
        "start_month_opt": "Model Controls",
        "peak_month_opt": "Model Controls",
        "manager_start_month_opt": "Overhead and Management",
        "ops_manager_start_month_opt": "Overhead and Management",
        "marketing_manager_start_month_opt": "Overhead and Management",
        "raise_effective_month_opt": "Overhead and Management",
        "tech_staffing_events_editor": "Staffing and Ops",
        "sales_staffing_events_editor": "Staffing and Ops",
        "res_new_build_install_schedule_editor": "New-Build Installs",
        "lc_new_build_install_schedule_editor": "New-Build Installs",
    }
)


def _input_section_for_key(key: str) -> str:
    return INPUT_KEY_TO_SECTION.get(key, "Working Capital, Debt, and Distributions")


def _attach_section_on_change(key: str, kwargs: dict) -> dict:
    section = INPUT_KEY_TO_SECTION.get(str(key or ""))
    if not section:
        return kwargs
    user_on_change = kwargs.get("on_change")
    user_args = tuple(kwargs.get("args", ()))
    user_kwargs = dict(kwargs.get("kwargs", {}))

    def _wrapped_on_change() -> None:
        st.session_state["last_active_input_section"] = section
        if callable(user_on_change):
            user_on_change(*user_args, **user_kwargs)

    patched = dict(kwargs)
    patched["on_change"] = _wrapped_on_change
    patched["args"] = ()
    patched["kwargs"] = {}
    return patched


def _search_input_keys(query: str) -> list[tuple[str, str, str]]:
    q = str(query or "").strip().lower()
    if not q:
        return []
    matches: list[tuple[str, str, str]] = []
    for key in sorted(DEFAULTS.keys()):
        note = str(INPUT_GUIDANCE.get(key, {}).get("note", ""))
        if q in key.lower() or q in note.lower():
            matches.append((key, _input_section_for_key(key), note))
    return matches[:30]


def _changed_assumption_keys(current: dict, applied: dict) -> list[str]:
    changed: list[str] = []
    for key in DEFAULTS:
        if current.get(key) != applied.get(key):
            changed.append(key)
    return changed


def _pending_change_keys_from_state() -> list[str]:
    current = _assumptions_from_state()
    applied_json = st.session_state.get("applied_assumptions_json", "")
    if not applied_json:
        return []
    try:
        applied = json.loads(applied_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return _changed_assumption_keys(current, applied)


def _assumptions_from_state() -> dict:
    return {k: deepcopy(st.session_state.get(k, v)) for k, v in DEFAULTS.items()}


def _serialize_assumptions(assumptions: dict) -> str:
    return json.dumps(assumptions, sort_keys=True, separators=(",", ":"))


@st.cache_data(show_spinner=False)
def _run_model_cached(assumptions_json: str) -> pd.DataFrame:
    assumptions = json.loads(assumptions_json)
    return run_model(assumptions)


@st.cache_data(show_spinner=False)
def _run_sensitivity_cached(assumptions_json: str, delta: float, drivers: tuple[str, ...]) -> tuple[pd.DataFrame, int]:
    assumptions = json.loads(assumptions_json)
    return run_one_way_sensitivity(assumptions, delta, drivers=list(drivers))


def _apply_helper_shifts(revenue_shift_pct: float, cost_shift_pct: float) -> None:
    rev_mult = 1 + revenue_shift_pct
    cost_mult = 1 + cost_shift_pct
    revenue_keys = [
        "avg_service_ticket",
        "avg_repl_ticket",
        "res_maint_monthly_fee",
        "lc_maint_quarterly_fee",
        "res_new_build_avg_ticket",
        "lc_new_build_avg_ticket",
        "res_service_upsell_revenue_per_visit",
        "lc_service_upsell_revenue_per_visit",
        "res_maint_upsell_revenue_per_visit",
        "lc_maint_upsell_revenue_per_visit",
    ]
    cost_keys = [
        "tech_wage_per_hour",
        "sales_wage_per_hour",
        "cost_per_lead",
        "permit_cost_per_repl_job",
        "disposal_cost_per_repl_job",
        "res_cost_per_maint_visit",
        "lc_cost_per_maint_visit",
        "fuel_per_truck_monthly",
        "maint_per_truck_monthly",
        "truck_insurance_per_truck_monthly",
        "office_payroll_monthly",
        "rent_monthly",
        "utilities_monthly",
        "insurance_monthly",
        "software_monthly",
        "other_fixed_monthly",
    ]
    for key in revenue_keys:
        if key in st.session_state:
            st.session_state[key] = max(0.0, float(st.session_state[key]) * rev_mult)
    for key in cost_keys:
        if key in st.session_state:
            st.session_state[key] = max(0.0, float(st.session_state[key]) * cost_mult)


def _apply_assumptions_to_state(assumptions: dict) -> None:
    migrated, _, _ = migrate_assumptions(assumptions)
    for k, v in migrated.items():
        st.session_state[k] = deepcopy(v)
    st.session_state["tech_staffing_events_master"] = deepcopy(migrated.get("tech_staffing_events", []))
    st.session_state["sales_staffing_events_master"] = deepcopy(migrated.get("sales_staffing_events", []))
    st.session_state["res_new_build_install_schedule_master"] = deepcopy(migrated.get("res_new_build_install_schedule", []))
    st.session_state["lc_new_build_install_schedule_master"] = deepcopy(migrated.get("lc_new_build_install_schedule", []))
    st.session_state.pop("tech_staffing_events_editor", None)
    st.session_state.pop("sales_staffing_events_editor", None)
    st.session_state.pop("res_new_build_install_schedule_editor", None)
    st.session_state.pop("lc_new_build_install_schedule_editor", None)
    st.session_state["applied_assumptions_json"] = _serialize_assumptions(_assumptions_from_state())


def _events_df(events: list[dict]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=["month", "hires", "attrition"])
    return pd.DataFrame(events)[["month", "hires", "attrition"]]


def _events_from_editor(df: pd.DataFrame) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    out = []
    for _, row in df.iterrows():
        month = str(row.get("month", "")).strip()
        if not month:
            continue
        try:
            hires = max(0, int(row.get("hires", 0)))
            attrition = max(0, int(row.get("attrition", 0)))
        except (TypeError, ValueError):
            continue
        out.append({"month": month, "hires": hires, "attrition": attrition})
    return _coalesce_events_by_month(out)


def _coalesce_events_by_month(events: list[dict]) -> list[dict]:
    if not events:
        return []
    merged: dict[str, dict] = {}
    for row in events:
        month = str(row.get("month", "")).strip()
        if not month:
            continue
        item = merged.setdefault(month, {"month": month, "hires": 0, "attrition": 0})
        item["hires"] += max(0, int(row.get("hires", 0)))
        item["attrition"] += max(0, int(row.get("attrition", 0)))
    return _sort_events(list(merged.values()))


def _sort_events(events: list[dict]) -> list[dict]:
    return sorted(
        events,
        key=lambda r: (str(r.get("month", "")), int(r.get("hires", 0)), int(r.get("attrition", 0))),
    )


def _partition_events_by_horizon(events: list[dict], month_labels: list[str]) -> tuple[list[dict], list[dict]]:
    allowed = set(month_labels)
    in_horizon = [e for e in events if str(e.get("month", "")) in allowed]
    out_of_horizon = [e for e in events if str(e.get("month", "")) not in allowed]
    return _sort_events(in_horizon), _sort_events(out_of_horizon)


def _merge_event_sets(in_horizon_events: list[dict], out_of_horizon_events: list[dict]) -> list[dict]:
    return _sort_events(in_horizon_events + out_of_horizon_events)


def _schedule_df(schedule: list[dict]) -> pd.DataFrame:
    if not schedule:
        return pd.DataFrame(columns=["month", "installs"])
    return pd.DataFrame(schedule)[["month", "installs"]]


def _schedule_from_editor(df: pd.DataFrame) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    out = []
    for _, row in df.iterrows():
        month = str(row.get("month", "")).strip()
        if not month:
            continue
        try:
            installs = max(0.0, float(row.get("installs", 0.0)))
        except (TypeError, ValueError):
            continue
        out.append({"month": month, "installs": installs})
    return _coalesce_schedule_by_month(out)


def _coalesce_schedule_by_month(schedule: list[dict]) -> list[dict]:
    if not schedule:
        return []
    merged: dict[str, dict] = {}
    for row in schedule:
        month = str(row.get("month", "")).strip()
        if not month:
            continue
        item = merged.setdefault(month, {"month": month, "installs": 0.0})
        item["installs"] += max(0.0, float(row.get("installs", 0.0)))
    return _sort_schedule(list(merged.values()))


def _sort_schedule(schedule: list[dict]) -> list[dict]:
    return sorted(
        schedule,
        key=lambda r: (str(r.get("month", "")), float(r.get("installs", 0.0))),
    )


def _partition_schedule_by_horizon(schedule: list[dict], month_labels: list[str]) -> tuple[list[dict], list[dict]]:
    allowed = set(month_labels)
    in_horizon = [r for r in schedule if str(r.get("month", "")) in allowed]
    out_of_horizon = [r for r in schedule if str(r.get("month", "")) not in allowed]
    return _sort_schedule(in_horizon), _sort_schedule(out_of_horizon)


def _merge_schedule_sets(in_horizon_schedule: list[dict], out_of_horizon_schedule: list[dict]) -> list[dict]:
    return _sort_schedule(in_horizon_schedule + out_of_horizon_schedule)


def _build_annual_kpis(df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    annual_kpis = (
        metrics["revenue_by_year"]
        .merge(metrics["ebitda_by_year"], on="Year")
        .merge(metrics["fcf_by_year"], on="Year")
    )
    annual_kpis["EBITDA Margin %"] = 100 * annual_kpis["EBITDA"] / annual_kpis["Total Revenue"].replace(0, pd.NA)
    annual_kpis["DSCR"] = metrics["dscr_by_year"].reset_index(drop=True)
    annual_kpis["Revenue per Tech"] = metrics["revenue_per_tech_by_year"].reset_index(drop=True)
    annual_kpis["Revenue per Truck"] = metrics["revenue_per_truck_by_year"].reset_index(drop=True)
    months_per_year = df.groupby("Year", as_index=False).size().rename(columns={"size": "Months in Period"})
    annual_kpis = annual_kpis.merge(months_per_year, on="Year", how="left")
    annual_kpis["Period Type"] = annual_kpis["Months in Period"].apply(
        lambda m: "Full Year" if int(m) == 12 else f"Partial ({int(m)} months)"
    )
    annual_kpis["Months in Period"] = annual_kpis["Months in Period"].astype(int)
    return annual_kpis.round(
        {
            "Total Revenue": 0,
            "EBITDA": 0,
            "Free Cash Flow": 0,
            "EBITDA Margin %": 2,
            "DSCR": 2,
            "Revenue per Tech": 0,
            "Revenue per Truck": 0,
        }
    )


def _current_ui_state() -> dict:
    charts = []
    for slot in range(1, 5):
        charts.append(
            {
                "enabled": bool(st.session_state.get(f"chart_{slot}_enabled", False)),
                "chart_type": st.session_state.get(f"chart_{slot}_type", "line"),
                "columns": st.session_state.get(f"chart_{slot}_cols", []),
                "title": st.session_state.get(f"chart_{slot}_title", f"Custom Chart {slot}"),
            }
        )
    return {
        "preset_scenarios": deepcopy(st.session_state.get("preset_scenarios", PRESET_SCENARIOS)),
        "input_focus_section": st.session_state.get("input_focus_section", "All Sections"),
        "expand_all_input_sections": st.session_state.get("expand_all_input_sections", False),
        "range_preset": st.session_state.get("range_preset", "Full horizon"),
        "range_start_label": st.session_state.get("range_start_label", ""),
        "range_end_label": st.session_state.get("range_end_label", ""),
        "value_mode": st.session_state.get("value_mode", "nominal"),
        "table_selected_columns": st.session_state.get("table_selected_columns", []),
        "sensitivity_delta": st.session_state.get("sensitivity_delta", 0.1),
        "sensitivity_drivers": st.session_state.get("sensitivity_drivers", []),
        "auto_run_model": st.session_state.get("auto_run_model", True),
        "custom_charts": charts,
        "goal_seek": {
            "metric": st.session_state.get("goal_target_metric", "Total EBITDA"),
            "input": st.session_state.get("goal_adjustable_input", "avg_service_ticket"),
            "target_value": st.session_state.get("goal_target_value", 0.0),
            "column_agg": st.session_state.get("goal_column_agg", "sum"),
            "column_name": st.session_state.get("goal_column_name", "EBITDA"),
            "year": st.session_state.get("goal_year", 1),
        },
    }


def _queue_deferred_state_update(key: str, value) -> None:
    pending = st.session_state.get("_deferred_session_state_updates")
    if not isinstance(pending, dict):
        pending = {}
    pending[key] = deepcopy(value)
    st.session_state["_deferred_session_state_updates"] = pending


def _set_session_state_or_defer(key: str, value) -> bool:
    try:
        st.session_state[key] = deepcopy(value)
        return False
    except StreamlitAPIException:
        _queue_deferred_state_update(key, value)
        return True


def _apply_deferred_state_updates() -> None:
    pending = st.session_state.get("_deferred_session_state_updates")
    if not isinstance(pending, dict) or not pending:
        return
    st.session_state["_deferred_session_state_updates"] = {}
    for key, value in pending.items():
        try:
            st.session_state[key] = deepcopy(value)
        except StreamlitAPIException:
            _queue_deferred_state_update(key, value)


def _apply_ui_state(ui_state: dict) -> None:
    if not isinstance(ui_state, dict):
        return
    if "preset_scenarios" in ui_state and isinstance(ui_state["preset_scenarios"], dict):
        migrated_presets: dict[str, dict] = {}
        for name, preset in ui_state["preset_scenarios"].items():
            if not isinstance(preset, dict):
                continue
            migrated, _, _ = migrate_assumptions(preset)
            migrated_presets[str(name)] = migrated
        if migrated_presets:
            st.session_state["preset_scenarios"] = migrated_presets
    if "preset_scenarios" not in st.session_state or not isinstance(st.session_state.get("preset_scenarios"), dict):
        st.session_state["preset_scenarios"] = deepcopy(PRESET_SCENARIOS)
    for preset_name, preset_values in PRESET_SCENARIOS.items():
        st.session_state["preset_scenarios"].setdefault(preset_name, deepcopy(preset_values))

    for key in [
        "input_focus_section",
        "expand_all_input_sections",
        "range_preset",
        "range_start_label",
        "range_end_label",
        "table_selected_columns",
        "sensitivity_delta",
        "sensitivity_drivers",
        "auto_run_model",
    ]:
        if key in ui_state:
            _set_session_state_or_defer(key, ui_state[key])
    if "value_mode" in ui_state:
        _set_session_state_or_defer("value_mode", ui_state["value_mode"])

    charts = ui_state.get("custom_charts", [])
    if isinstance(charts, list):
        for slot, cfg in enumerate(charts[:4], start=1):
            if not isinstance(cfg, dict):
                continue
            _set_session_state_or_defer(f"chart_{slot}_enabled", bool(cfg.get("enabled", False)))
            _set_session_state_or_defer(f"chart_{slot}_type", cfg.get("chart_type", "line"))
            _set_session_state_or_defer(f"chart_{slot}_cols", cfg.get("columns", []))
            _set_session_state_or_defer(f"chart_{slot}_title", cfg.get("title", f"Custom Chart {slot}"))

    goal = ui_state.get("goal_seek", {})
    if isinstance(goal, dict):
        _set_session_state_or_defer(
            "goal_target_metric", goal.get("metric", st.session_state.get("goal_target_metric", "Total EBITDA"))
        )
        _set_session_state_or_defer(
            "goal_adjustable_input", goal.get("input", st.session_state.get("goal_adjustable_input", "avg_service_ticket"))
        )
        _set_session_state_or_defer(
            "goal_target_value", goal.get("target_value", st.session_state.get("goal_target_value", 0.0))
        )
        _set_session_state_or_defer("goal_column_agg", goal.get("column_agg", st.session_state.get("goal_column_agg", "sum")))
        _set_session_state_or_defer("goal_column_name", goal.get("column_name", st.session_state.get("goal_column_name", "EBITDA")))
        _set_session_state_or_defer("goal_year", goal.get("year", st.session_state.get("goal_year", 1)))


def _metric_from_selection(
    df: pd.DataFrame,
    metric_name: str,
    column_name: str,
    agg: str,
    year: int,
) -> float:
    if metric_name == "Total Revenue":
        return float(df["Total Revenue"].sum())
    if metric_name == "Total EBITDA":
        return float(df["EBITDA"].sum())
    if metric_name == "Total Free Cash Flow":
        return float(df["Free Cash Flow"].sum())
    if metric_name == "Minimum Ending Cash":
        return float(df["End Cash"].min())
    if metric_name == "Negative Cash Months":
        return float((df["End Cash"] < 0).sum())
    if metric_name == "Avg Gross Margin %":
        rev = float(df["Total Revenue"].sum())
        gp = float(df["Gross Profit"].sum())
        return 100.0 * gp / rev if rev else 0.0
    if metric_name == "Year Total Revenue":
        return float(df.loc[df["Year"] == year, "Total Revenue"].sum())
    if metric_name == "Year EBITDA":
        return float(df.loc[df["Year"] == year, "EBITDA"].sum())
    if metric_name == "Year Free Cash Flow":
        return float(df.loc[df["Year"] == year, "Free Cash Flow"].sum())
    if metric_name == "Column Aggregate":
        series = df[column_name]
        if agg == "sum":
            return float(series.sum())
        if agg == "avg":
            return float(series.mean())
        if agg == "min":
            return float(series.min())
        if agg == "max":
            return float(series.max())
        if agg == "end":
            return float(series.iloc[-1])
    return 0.0


def _filter_df(df: pd.DataFrame, start_label: str, end_label: str) -> pd.DataFrame:
    labels = df["Year_Month_Label"].tolist()
    if start_label not in labels or end_label not in labels:
        return df
    start_idx = labels.index(start_label)
    end_idx = labels.index(end_label)
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    return df.iloc[start_idx : end_idx + 1]


def _stable_json(value) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


def _monthly_series_from_rows(month_labels: list[str], rows: list[dict], value_key: str) -> list[float]:
    by_month: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        month = str(row.get("month", "")).strip()
        if month not in month_labels:
            continue
        try:
            value = float(row.get(value_key, 0.0))
        except (TypeError, ValueError):
            continue
        value = max(0.0, value)
        by_month[month] = by_month.get(month, 0.0) + value
    return [by_month.get(m, 0.0) for m in month_labels]


def _build_input_timeseries(assumptions: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    month_labels = [pd.Timestamp(d).strftime("%Y-%m") for d in dates]
    horizon = len(month_labels)
    columns: dict[str, list | pd.DatetimeIndex] = {
        "Date": pd.DatetimeIndex(dates),
        "Year_Month_Label": month_labels,
    }
    for key in sorted(DEFAULTS.keys()):
        value = assumptions.get(key, DEFAULTS[key])
        if isinstance(value, bool):
            scalar = int(value)
        elif isinstance(value, (int, float, str)):
            scalar = value
        else:
            scalar = _stable_json(value)
        columns[key] = [scalar] * horizon

    tech_events = assumptions.get("tech_staffing_events", [])
    sales_events = assumptions.get("sales_staffing_events", [])
    res_schedule = assumptions.get("res_new_build_install_schedule", [])
    lc_schedule = assumptions.get("lc_new_build_install_schedule", [])

    columns["tech_staffing_events_hires_input"] = _monthly_series_from_rows(month_labels, tech_events, "hires")
    columns["tech_staffing_events_attrition_input"] = _monthly_series_from_rows(month_labels, tech_events, "attrition")
    columns["sales_staffing_events_hires_input"] = _monthly_series_from_rows(month_labels, sales_events, "hires")
    columns["sales_staffing_events_attrition_input"] = _monthly_series_from_rows(month_labels, sales_events, "attrition")
    columns["res_new_build_install_schedule_installs_input"] = _monthly_series_from_rows(month_labels, res_schedule, "installs")
    columns["lc_new_build_install_schedule_installs_input"] = _monthly_series_from_rows(month_labels, lc_schedule, "installs")

    return pd.DataFrame(columns)


def _display_help_panel() -> None:
    with st.expander("Assumption Guidance and Reasonable Ranges", expanded=False):
        query = st.text_input("Filter guidance", placeholder="Search by input key or topic")
        rows = []
        for key, g in INPUT_GUIDANCE.items():
            rows.append({"Input": key, "Range": f"{g['min']} to {g['max']}", "Guidance": g["note"]})
        guidance_df = pd.DataFrame(rows)
        if query:
            q = query.strip().lower()
            guidance_df = guidance_df[
                guidance_df["Input"].str.lower().str.contains(q, regex=False)
                | guidance_df["Guidance"].str.lower().str.contains(q, regex=False)
            ]
        st.dataframe(_format_dataframe_for_display(guidance_df), width="stretch", hide_index=True)


def _sensitivity_objective_sign(target_metric: str) -> float:
    # Most targets are "higher is better"; disbursements is "lower is better".
    return -1.0 if target_metric == "Total Disbursements" else 1.0


def _build_sensitivity_insights(sens_df: pd.DataFrame, target_metric: str, delta_pct: float) -> pd.DataFrame:
    delta_col = f"Delta {target_metric}"
    if sens_df is None or len(sens_df) == 0 or delta_col not in sens_df.columns:
        return pd.DataFrame()

    pivot = sens_df.pivot_table(index="Driver", columns="Case", values=delta_col, aggfunc="mean").fillna(0.0)
    if "Low" not in pivot.columns:
        pivot["Low"] = 0.0
    if "High" not in pivot.columns:
        pivot["High"] = 0.0

    sign = _sensitivity_objective_sign(target_metric)
    rows: list[dict] = []
    scale = max(float(delta_pct) * 100.0, 1e-9)
    for driver, row in pivot.iterrows():
        low_delta = float(row["Low"])
        high_delta = float(row["High"])
        low_score = sign * low_delta
        high_score = sign * high_delta

        if high_score >= low_score:
            focus_direction = "Increase input"
            focus_case = "High"
            focus_metric_delta = high_delta
            focus_value_gain = max(0.0, high_score)
        else:
            focus_direction = "Decrease input"
            focus_case = "Low"
            focus_metric_delta = low_delta
            focus_value_gain = max(0.0, low_score)

        best_gain = max(0.0, low_score, high_score)
        leakage_risk = max(0.0, -min(low_score, high_score))
        sensitivity_abs = max(abs(low_score), abs(high_score))

        if sensitivity_abs == 0:
            posture = "Low materiality"
        elif leakage_risk > best_gain * 1.25:
            posture = "Leakage control priority"
        elif best_gain > leakage_risk * 1.25:
            posture = "Value growth priority"
        else:
            posture = "Balanced with guardrails"

        rows.append(
            {
                "Driver": driver,
                "Low Delta": low_delta,
                "High Delta": high_delta,
                "Value Gain Potential": best_gain,
                "Leakage Risk": leakage_risk,
                "Net Opportunity": best_gain - leakage_risk,
                "Focus Direction": focus_direction,
                "Focus Case": focus_case,
                "Focus Delta (Target)": focus_metric_delta,
                "Sensitivity (abs)": sensitivity_abs,
                "Sensitivity per 1% Input Move": sensitivity_abs / scale,
                "Posture": posture,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Sensitivity (abs)", "Value Gain Potential"], ascending=[False, False]).reset_index(drop=True)


def _format_money_like(value: float) -> str:
    return f"${value:,.0f}"


PCT_KEY_TOKENS = ("pct", "rate", "margin", "penetration", "conversion", "amplitude")
CURRENCY_KEY_TOKENS = (
    "revenue",
    "ticket",
    "wage",
    "cost",
    "salary",
    "payment",
    "principal",
    "limit",
    "cash",
    "balance",
    "capex",
    "fee",
    "spend",
    "insurance",
    "rent",
    "utilities",
    "software",
    "payroll",
    "disposal",
    "permit",
    "lead",
    "draw",
    "repay",
    "distribution",
    "disbursement",
    "ebitda",
    "profit",
    "fcf",
    "loan",
    "loc",
    "amount",
)

MODEL_KEY_ALIASES = {
    "start_month_opt": "start_month",
    "peak_month_opt": "peak_month",
    "manager_start_month_opt": "manager_start_month",
    "ops_manager_start_month_opt": "ops_manager_start_month",
    "marketing_manager_start_month_opt": "marketing_manager_start_month",
    "raise_effective_month_opt": "raise_effective_month",
}

UI_STATE_KEY_PREFIXES = (
    "chart_",
    "ai_export_",
    "template_",
    "input_",
    "runtime_",
    "range_",
    "table_",
    "sensitivity_",
    "goal_",
    "helper_",
)

WIDGET_HELP_OVERRIDES_BY_KEY = {
    "start_month_opt": "Choose the first calendar month of the projection timeline.",
    "peak_month_opt": "Select the highest-demand month used to shape seasonality.",
    "manager_start_month_opt": "Choose the month when manager salary starts being recognized.",
    "ops_manager_start_month_opt": "Choose the month when ops-manager salary starts being recognized.",
    "marketing_manager_start_month_opt": "Choose the month when marketing-manager salary starts being recognized.",
    "raise_effective_month_opt": "Sets the month when annual wage/salary raises take effect each year.",
    "input_search_query": "Search inputs by name or guidance text to jump to the right section quickly.",
    "input_match_choice": "Pick a matched input to focus that section and review the guidance note.",
    "template_apply_scenario_name": "Choose one parsed template scenario to apply immediately to current assumptions.",
    "table_selected_columns": "Select which columns appear in the line-by-line cashflow table and CSV export.",
    "input_ts_scope": "Switch input validation charts between the selected range and full model horizon.",
    "input_ts_selected_cols": "Pick one or more input series to validate assumptions over time.",
    "runtime_log_limit": "Control how many recent runtime events are shown for diagnostics.",
}

WIDGET_HELP_OVERRIDES_BY_LABEL = {
    "save name": "Enter a unique, descriptive name so scenarios/workspaces are easy to find later.",
    "saved scenarios": "Choose a saved scenario to load, delete, or reuse as a preset source.",
    "saved workspaces": "Choose a saved workspace to load, delete, or include in exports.",
    "displayed table columns": "Select the columns shown in the cashflow table and export.",
    "select month for cash flow bridge": "Choose a month to inspect its detailed operating, investing, and financing bridge.",
    "matching inputs": "Choose a match to navigate directly to that input's section.",
}

_ORIG_NUMBER_INPUT = st.number_input
_ORIG_SLIDER = st.slider
_ORIG_SELECTBOX = st.selectbox
_ORIG_TOGGLE = st.toggle
_ORIG_CHECKBOX = st.checkbox
_ORIG_MULTISELECT = st.multiselect
_ORIG_DATA_EDITOR = st.data_editor
_ORIG_TEXT_INPUT = st.text_input
_ORIG_TEXT_AREA = st.text_area
_ORIG_RADIO = st.radio
_ORIG_BUTTON = st.button
_ORIG_DOWNLOAD_BUTTON = st.download_button
_ORIG_FILE_UPLOADER = st.file_uploader


def _decimals_from_step(step: float | int | None, fallback: int = 2) -> int:
    if step is None:
        return fallback
    try:
        s = abs(float(step))
    except (TypeError, ValueError):
        return fallback
    if s >= 1:
        return 0
    txt = f"{s:.10f}".rstrip("0")
    if "." not in txt:
        return fallback
    return min(6, len(txt.split(".")[1]))


def _is_pct_field(key: str, label: str, min_value, max_value) -> bool:
    k = str(key or "").lower()
    l = str(label or "").lower()
    if any(tok in k for tok in PCT_KEY_TOKENS) or "%" in l:
        return True
    if isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)):
        return float(min_value) >= 0 and float(max_value) <= 1
    return False


def _is_currency_field(key: str, label: str) -> bool:
    k = str(key or "").lower()
    l = str(label or "").lower()
    return any(tok in k for tok in CURRENCY_KEY_TOKENS) or "$" in l


def _is_number_like(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt_help_number(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return f"{value:,}"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(num - round(num)) < 1e-9:
        return f"{int(round(num)):,}"
    if abs(num) >= 100:
        return f"{num:,.2f}".rstrip("0").rstrip(".")
    if abs(num) >= 1:
        return f"{num:.3f}".rstrip("0").rstrip(".")
    return f"{num:.4f}".rstrip("0").rstrip(".")


def _normalized_label(label: str) -> str:
    return " ".join(str(label or "").split()).strip().lower()


def _resolve_model_key_for_help(key: str) -> str | None:
    key_text = str(key or "").strip()
    if not key_text:
        return None
    if key_text in DEFAULTS:
        return key_text
    alias = MODEL_KEY_ALIASES.get(key_text)
    if alias in DEFAULTS:
        return alias
    if key_text.endswith("_opt"):
        candidate = key_text[:-4]
        if candidate in DEFAULTS:
            return candidate
    return None


def _is_ui_state_key(key: str) -> bool:
    key_text = str(key or "")
    if not key_text:
        return False
    if key_text in UI_DEFAULTS:
        return True
    if key_text.startswith(UI_STATE_KEY_PREFIXES):
        return True
    return False


def _coerce_options_preview(options, max_items: int = 6) -> tuple[list[str], int]:
    if options is None:
        return [], 0
    try:
        option_list = list(options)
    except TypeError:
        return [], 0
    count = len(option_list)
    if count == 0:
        return [], 0
    preview = []
    for item in option_list[:max_items]:
        txt = str(item).strip()
        preview.append(txt if len(txt) <= 36 else f"{txt[:33]}...")
    return preview, count


def _widget_usage_hint(widget_type: str, label: str) -> str:
    label_l = _normalized_label(label)
    if widget_type == "number_input":
        return "Enter a numeric value for precise what-if edits."
    if widget_type == "slider":
        return "Drag to test scenarios quickly."
    if widget_type in {"selectbox", "radio"}:
        return "Choose the option that best matches your operating plan."
    if widget_type == "multiselect":
        return "Select one or more options."
    if widget_type in {"toggle", "checkbox"}:
        return "Turn on to enable; turn off to disable."
    if widget_type == "text_input":
        if any(tok in label_l for tok in ("search", "find", "filter")):
            return "Type keywords to narrow the list quickly."
        if any(tok in label_l for tok in ("name", "title")):
            return "Use a clear label so saved items are easy to identify."
        return "Type the text value for this setting."
    if widget_type == "text_area":
        return "Use this field for longer text content."
    if widget_type == "button":
        return "Click to run this action."
    if widget_type == "download_button":
        return "Click to download the prepared file."
    if widget_type == "file_uploader":
        return "Upload a supported file, then apply it in the workflow."
    return ""


def _widget_impact_hint(model_key: str | None, key: str, label: str, widget_type: str) -> str:
    text = f"{model_key or ''} {key or ''} {label or ''}".lower()
    if widget_type == "download_button":
        return "Impact: exports a snapshot only; model calculations do not change."
    if widget_type == "file_uploader":
        return "Impact: uploaded data can overwrite assumptions when applied."
    if widget_type == "button":
        if any(tok in text for tok in ("delete", "clear", "reset", "restore")):
            return "Impact: may remove or reset saved/app state immediately."
        if any(tok in text for tok in ("save", "load", "apply", "run", "generate", "import")):
            return "Impact: updates app state and can change outputs."
        return "Impact: performs the selected workflow action."

    if model_key is None:
        if _is_ui_state_key(key) or any(tok in text for tok in ("chart", "column", "scope", "search", "filter", "title")):
            return "Impact: affects UI/workflow only; core cashflow math is unchanged."
        return "Impact: updates this application control."

    if str(model_key).startswith("enable_") or widget_type in {"toggle", "checkbox"}:
        return "Impact: turns this model feature on/off and can materially change outputs."
    if any(tok in text for tok in ("horizon", "start_year", "start_month", "peak_month", "seasonality", "effective_month")):
        return "Impact: changes projection timing and seasonal shape."
    if any(
        tok in text
        for tok in (
            "loan",
            "loc",
            "distributions",
            "cash_target",
            "starting_cash",
            "principal",
            "repay",
            "draw",
            "interest",
        )
    ):
        return "Impact: changes financing cashflows, liquidity, and runway."
    if any(tok in text for tok in ("ar_days", "ap_days", "inventory_days", "opening_ar_balance", "opening_ap_balance", "opening_inventory_balance")):
        return "Impact: changes working-capital timing and cash conversion."
    if "churn" in text:
        return "Impact: higher churn reduces retained agreements and recurring revenue."
    if any(tok in text for tok in ("discount_rate", "value_mode")):
        return "Impact: changes valuation/display interpretation more than operating math."
    if any(
        tok in text
        for tok in (
            "cost",
            "wage",
            "salary",
            "payroll",
            "rent",
            "utilities",
            "insurance",
            "software",
            "fuel",
            "material",
            "equipment",
            "burden",
            "payment",
            "capex",
        )
    ):
        return "Impact: higher values usually increase costs and reduce EBITDA/cash."
    if any(
        tok in text
        for tok in (
            "revenue",
            "ticket",
            "lead",
            "call",
            "tech",
            "sales",
            "install",
            "agreement",
            "close",
            "conversion",
            "upsell",
            "price",
            "capacity",
            "fee",
        )
    ):
        return "Impact: higher values usually increase demand/revenue potential, with related cost effects."
    if "mode" in text:
        return "Impact: switches calculation logic and can materially change outputs."
    return "Impact: directly changes projected revenue, costs, and cash flow."


def _widget_range_hint(
    *,
    model_key: str | None,
    key: str,
    label: str,
    widget_type: str,
    min_value,
    max_value,
    options,
    file_types,
) -> str:
    if model_key and model_key in INPUT_GUIDANCE:
        g = INPUT_GUIDANCE[model_key]
        note = str(g.get("note", "")).strip()
        text = f"Realistic range: {_fmt_help_number(g['min'])} to {_fmt_help_number(g['max'])}."
        if note:
            text = f"{text} {note}"
        return text

    if widget_type in {"selectbox", "radio", "multiselect"}:
        preview, count = _coerce_options_preview(options)
        if count > 0:
            if count <= len(preview):
                return f"Options: {', '.join(preview)}."
            return f"{count} options available. Examples: {', '.join(preview)}."

    if widget_type == "file_uploader":
        if isinstance(file_types, str):
            extensions = [file_types]
        else:
            try:
                extensions = list(file_types or [])
            except TypeError:
                extensions = []
        extensions = [f".{str(ext).lstrip('.')}" for ext in extensions if str(ext).strip()]
        if extensions:
            return f"Accepted file types: {', '.join(extensions)}."

    has_min = _is_number_like(min_value)
    has_max = _is_number_like(max_value)
    if has_min and has_max:
        hint = f"UI range: {_fmt_help_number(min_value)} to {_fmt_help_number(max_value)}."
        if _is_pct_field(str(model_key or key), label, min_value, max_value):
            hint = f"{hint} (0 to 1 maps to 0% to 100%)."
        return hint
    if has_min:
        return f"UI minimum: {_fmt_help_number(min_value)}."
    if has_max:
        return f"UI maximum: {_fmt_help_number(max_value)}."

    baseline_key = model_key if model_key in DEFAULTS else str(key or "")
    baseline = DEFAULTS.get(baseline_key)
    if _is_number_like(baseline):
        if _is_pct_field(str(model_key or key), label, None, None):
            return f"Current baseline: {_fmt_help_number(baseline)} ({float(baseline) * 100:.1f}%)."
        return f"Current baseline: {_fmt_help_number(baseline)}."
    return ""


def _base_widget_help(label: str, key: str, model_key: str | None, widget_type: str) -> str:
    key_text = str(key or "").strip()
    label_text = " ".join(str(label or "").split()).strip()
    label_lookup = _normalized_label(label_text)

    if key_text in WIDGET_HELP_OVERRIDES_BY_KEY:
        return WIDGET_HELP_OVERRIDES_BY_KEY[key_text]
    if model_key and model_key in WIDGET_HELP_OVERRIDES_BY_KEY:
        return WIDGET_HELP_OVERRIDES_BY_KEY[model_key]
    if label_lookup in WIDGET_HELP_OVERRIDES_BY_LABEL:
        return WIDGET_HELP_OVERRIDES_BY_LABEL[label_lookup]

    if widget_type == "button":
        if label_text:
            return f"Runs the '{label_text}' workflow action."
        return "Runs the selected workflow action."
    if widget_type == "download_button":
        return "Downloads the prepared output file."
    if widget_type == "file_uploader":
        return "Uploads external data into the app workflow."

    if model_key:
        return f"{label_text or model_key} controls the `{model_key}` assumption."
    if key_text:
        return f"{label_text or key_text} controls this application setting."
    if label_text:
        return f"{label_text} controls this interaction."
    return ""


def _build_widget_help(
    *,
    label: str,
    key: str,
    widget_type: str,
    min_value=None,
    max_value=None,
    options=None,
    file_types=None,
) -> str:
    model_key = _resolve_model_key_for_help(key)
    parts = [
        _base_widget_help(label, key, model_key, widget_type),
        _widget_usage_hint(widget_type, label),
        _widget_impact_hint(model_key, key, label, widget_type),
        _widget_range_hint(
            model_key=model_key,
            key=key,
            label=str(label or ""),
            widget_type=widget_type,
            min_value=min_value,
            max_value=max_value,
            options=options,
            file_types=file_types,
        ),
    ]
    merged = " ".join(str(part).strip() for part in parts if str(part).strip())
    merged = " ".join(merged.split())
    if len(merged) > 520:
        return f"{merged[:517].rstrip()}..."
    return merged


def _inject_auto_help(
    *,
    label: str,
    widget_type: str,
    args: tuple,
    kwargs: dict,
) -> dict:
    existing_help = kwargs.get("help")
    if isinstance(existing_help, str) and existing_help.strip():
        return kwargs
    if existing_help is not None and not isinstance(existing_help, str):
        return kwargs

    key = str(kwargs.get("key", "") or "")
    min_value = kwargs.get("min_value")
    max_value = kwargs.get("max_value")
    options = kwargs.get("options")
    file_types = kwargs.get("type")

    if widget_type == "number_input":
        if min_value is None and len(args) > 0:
            min_value = args[0]
        if max_value is None and len(args) > 1:
            max_value = args[1]
    elif widget_type == "slider":
        if min_value is None and len(args) > 0:
            min_value = args[0]
        if max_value is None and len(args) > 1:
            max_value = args[1]
    elif widget_type in {"selectbox", "radio", "multiselect"}:
        if options is None and len(args) > 0:
            options = args[0]
    elif widget_type == "file_uploader":
        if file_types is None and len(args) > 0:
            file_types = args[0]

    help_text = _build_widget_help(
        label=str(label or ""),
        key=key,
        widget_type=widget_type,
        min_value=min_value,
        max_value=max_value,
        options=options,
        file_types=file_types,
    )
    if not help_text:
        return kwargs
    patched = dict(kwargs)
    patched["help"] = help_text
    return patched


def _coerce_numeric(value, *, as_int: bool):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value) if as_int else float(int(value))
    if isinstance(value, (int, float)):
        return int(value) if as_int else float(value)
    if isinstance(value, str):
        txt = value.strip().replace(",", "").replace("$", "").replace("%", "")
        if txt == "":
            return None
        try:
            num = float(txt)
            return int(num) if as_int else float(num)
        except ValueError:
            return None
    return None


def _infer_int_mode(default_val, min_value, max_value, step, value_hint) -> bool:
    if not (isinstance(default_val, int) and not isinstance(default_val, bool)):
        return False
    for v in (min_value, max_value, step, value_hint):
        if isinstance(v, float):
            return False
        if isinstance(v, str):
            parsed = _coerce_numeric(v, as_int=False)
            if parsed is not None and abs(parsed - round(parsed)) > 1e-9:
                return False
    return True


def _ux_number_input(label, *args, **kwargs):
    key = kwargs.get("key", "")
    kwargs = _attach_section_on_change(key, kwargs)
    kwargs = _inject_auto_help(label=label, widget_type="number_input", args=args, kwargs=kwargs)
    if "format" in kwargs:
        return _ORIG_NUMBER_INPUT(label, *args, **kwargs)

    default_val = DEFAULTS.get(key)
    min_value = kwargs.get("min_value")
    max_value = kwargs.get("max_value")
    step = kwargs.get("step")
    value_hint = kwargs.get("value")
    if value_hint is None and key:
        value_hint = st.session_state.get(key)

    is_int_field = _infer_int_mode(default_val, min_value, max_value, step, value_hint)
    if is_int_field:
        coerced_min = _coerce_numeric(min_value, as_int=True)
        coerced_max = _coerce_numeric(max_value, as_int=True)
        coerced_step = _coerce_numeric(step, as_int=True)
        if coerced_min is not None:
            kwargs["min_value"] = coerced_min
        if coerced_max is not None:
            kwargs["max_value"] = coerced_max
        kwargs["step"] = coerced_step if coerced_step is not None else 1
        if "value" in kwargs:
            coerced_value = _coerce_numeric(kwargs.get("value"), as_int=True)
            if coerced_value is not None:
                kwargs["value"] = coerced_value
        if key and isinstance(st.session_state.get(key), str):
            coerced_state_value = _coerce_numeric(st.session_state.get(key), as_int=True)
            if coerced_state_value is not None:
                st.session_state[key] = coerced_state_value
        kwargs.setdefault("format", "%d")
        return _ORIG_NUMBER_INPUT(label, *args, **kwargs)

    if step is None:
        if _is_pct_field(str(key), str(label), min_value, max_value):
            kwargs["step"] = 0.001
            step = 0.001
        else:
            base = abs(float(default_val)) if isinstance(default_val, (int, float)) else 0.0
            if base >= 1000:
                kwargs["step"] = 100.0
                step = 100.0
            elif base >= 100:
                kwargs["step"] = 10.0
                step = 10.0
            elif base >= 10:
                kwargs["step"] = 1.0
                step = 1.0
            else:
                kwargs["step"] = 0.1
                step = 0.1

    coerced_min = _coerce_numeric(min_value, as_int=False)
    coerced_max = _coerce_numeric(max_value, as_int=False)
    coerced_step = _coerce_numeric(kwargs.get("step"), as_int=False)
    if coerced_min is not None:
        kwargs["min_value"] = coerced_min
    if coerced_max is not None:
        kwargs["max_value"] = coerced_max
    kwargs["step"] = coerced_step if coerced_step is not None else 0.1
    step = kwargs["step"]
    if "value" in kwargs:
        coerced_value = _coerce_numeric(kwargs.get("value"), as_int=False)
        if coerced_value is not None:
            kwargs["value"] = coerced_value
    if key and isinstance(st.session_state.get(key), str):
        coerced_state_value = _coerce_numeric(st.session_state.get(key), as_int=False)
        if coerced_state_value is not None:
            st.session_state[key] = coerced_state_value

    if _is_pct_field(str(key), str(label), min_value, max_value):
        decimals = max(3, _decimals_from_step(step, fallback=3))
    elif _is_currency_field(str(key), str(label)):
        decimals = 0 if float(step) >= 1 else 2
    else:
        decimals = _decimals_from_step(step, fallback=2)
    kwargs["format"] = f"%.{decimals}f"
    return _ORIG_NUMBER_INPUT(label, *args, **kwargs)


def _ux_slider(label, *args, **kwargs):
    key = kwargs.get("key", "")
    kwargs = _attach_section_on_change(key, kwargs)
    kwargs = _inject_auto_help(label=label, widget_type="slider", args=args, kwargs=kwargs)
    if "format" in kwargs:
        return _ORIG_SLIDER(label, *args, **kwargs)

    default_val = DEFAULTS.get(key)
    min_value = kwargs.get("min_value", args[0] if len(args) > 0 else None)
    max_value = kwargs.get("max_value", args[1] if len(args) > 1 else None)
    step = kwargs.get("step", args[3] if len(args) > 3 else None)

    is_int_field = isinstance(default_val, int) and not isinstance(default_val, bool)
    if is_int_field and not isinstance(min_value, float) and not isinstance(max_value, float):
        kwargs.setdefault("format", "%d")
    else:
        if step is None:
            step = 0.001 if _is_pct_field(str(key), str(label), min_value, max_value) else 0.01
        decimals = _decimals_from_step(step, fallback=3 if _is_pct_field(str(key), str(label), min_value, max_value) else 2)
        kwargs["format"] = f"%.{decimals}f"
    return _ORIG_SLIDER(label, *args, **kwargs)


def _ux_selectbox(label, *args, **kwargs):
    key = kwargs.get("key", "")
    kwargs = _attach_section_on_change(key, kwargs)
    kwargs = _inject_auto_help(label=label, widget_type="selectbox", args=args, kwargs=kwargs)
    return _ORIG_SELECTBOX(label, *args, **kwargs)


def _ux_toggle(label, *args, **kwargs):
    key = kwargs.get("key", "")
    kwargs = _attach_section_on_change(key, kwargs)
    kwargs = _inject_auto_help(label=label, widget_type="toggle", args=args, kwargs=kwargs)
    return _ORIG_TOGGLE(label, *args, **kwargs)


def _ux_checkbox(label, *args, **kwargs):
    key = kwargs.get("key", "")
    kwargs = _attach_section_on_change(key, kwargs)
    kwargs = _inject_auto_help(label=label, widget_type="checkbox", args=args, kwargs=kwargs)
    return _ORIG_CHECKBOX(label, *args, **kwargs)


def _ux_multiselect(label, *args, **kwargs):
    key = kwargs.get("key", "")
    kwargs = _attach_section_on_change(key, kwargs)
    kwargs = _inject_auto_help(label=label, widget_type="multiselect", args=args, kwargs=kwargs)
    return _ORIG_MULTISELECT(label, *args, **kwargs)


def _ux_data_editor(*args, **kwargs):
    key = kwargs.get("key", "")
    kwargs = _attach_section_on_change(key, kwargs)
    return _ORIG_DATA_EDITOR(*args, **kwargs)


def _ux_text_input(label, *args, **kwargs):
    key = kwargs.get("key", "")
    kwargs = _attach_section_on_change(key, kwargs)
    kwargs = _inject_auto_help(label=label, widget_type="text_input", args=args, kwargs=kwargs)
    return _ORIG_TEXT_INPUT(label, *args, **kwargs)


def _ux_text_area(label, *args, **kwargs):
    key = kwargs.get("key", "")
    kwargs = _attach_section_on_change(key, kwargs)
    kwargs = _inject_auto_help(label=label, widget_type="text_area", args=args, kwargs=kwargs)
    return _ORIG_TEXT_AREA(label, *args, **kwargs)


def _ux_radio(label, *args, **kwargs):
    key = kwargs.get("key", "")
    kwargs = _attach_section_on_change(key, kwargs)
    kwargs = _inject_auto_help(label=label, widget_type="radio", args=args, kwargs=kwargs)
    return _ORIG_RADIO(label, *args, **kwargs)


def _ux_button(label, *args, **kwargs):
    kwargs = _inject_auto_help(label=label, widget_type="button", args=args, kwargs=kwargs)
    return _ORIG_BUTTON(label, *args, **kwargs)


def _ux_download_button(label, *args, **kwargs):
    kwargs = _inject_auto_help(label=label, widget_type="download_button", args=args, kwargs=kwargs)
    return _ORIG_DOWNLOAD_BUTTON(label, *args, **kwargs)


def _ux_file_uploader(label, *args, **kwargs):
    key = kwargs.get("key", "")
    kwargs = _attach_section_on_change(key, kwargs)
    kwargs = _inject_auto_help(label=label, widget_type="file_uploader", args=args, kwargs=kwargs)
    return _ORIG_FILE_UPLOADER(label, *args, **kwargs)


def _series_is_integerish(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    if not pd.api.types.is_numeric_dtype(series):
        return False
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    return bool(((s - s.round()).abs() < 1e-9).mean() > 0.95)


def _format_currency_value(x: float) -> str:
    if x < 0:
        return f"-${abs(x):,.0f}"
    return f"${x:,.0f}"


def _format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(lambda v: "—" if pd.isna(v) else ("TRUE" if bool(v) else "FALSE"))
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        col_l = str(col).lower()
        is_pct = "%" in str(col) or any(tok in col_l for tok in ("pct", "margin", "rate", "conversion"))
        is_currency = any(tok in col_l for tok in CURRENCY_KEY_TOKENS)
        is_intish = _series_is_integerish(out[col])

        if is_pct:
            out[col] = out[col].map(lambda v: "—" if pd.isna(v) else f"{float(v):,.2f}%")
        elif is_currency:
            out[col] = out[col].map(lambda v: "—" if pd.isna(v) else _format_currency_value(float(v)))
        elif is_intish:
            out[col] = out[col].map(lambda v: "—" if pd.isna(v) else f"{float(v):,.0f}")
        else:
            out[col] = out[col].map(lambda v: "—" if pd.isna(v) else f"{float(v):,.2f}")
    return out


st.number_input = _ux_number_input
st.slider = _ux_slider
st.selectbox = _ux_selectbox
st.toggle = _ux_toggle
st.checkbox = _ux_checkbox
st.multiselect = _ux_multiselect
st.data_editor = _ux_data_editor
st.text_input = _ux_text_input
st.text_area = _ux_text_area
st.radio = _ux_radio
st.button = _ux_button
st.download_button = _ux_download_button
st.file_uploader = _ux_file_uploader


def _normalize_inputs_for_runtime(assumptions: dict) -> tuple[dict, list[str]]:
    """Apply best-effort safety normalization so the model can continue running."""
    safe = deepcopy(assumptions)
    warnings: list[str] = []

    def _int_at_least(key: str, floor: int) -> None:
        try:
            val = int(safe.get(key, floor))
        except (TypeError, ValueError):
            val = floor
        if val < floor:
            warnings.append(f"{key} adjusted to minimum {floor}.")
            val = floor
        safe[key] = val

    _int_at_least("horizon_months", 1)
    _int_at_least("loan_term_months", 1)
    _int_at_least("start_month", 1)
    _int_at_least("peak_month", 1)
    _int_at_least("raise_effective_month", 1)

    for key in ["start_month", "peak_month", "raise_effective_month"]:
        if safe[key] > 12:
            safe[key] = 12
            warnings.append(f"{key} adjusted to maximum 12.")

    if int(safe.get("max_techs", 0)) < int(safe.get("starting_techs", 0)):
        safe["max_techs"] = int(safe["starting_techs"])
        warnings.append("max_techs was below starting_techs and was adjusted upward.")

    enum_defaults = {
        "paid_leads_mode": "per_tech",
        "capex_trucks_mode": "purchase_with_downpayment",
        "new_build_mode": "base_seasonal",
        "asset_expiry_mode": "release",
    }
    enum_options = {
        "paid_leads_mode": {"fixed", "per_tech"},
        "capex_trucks_mode": {"payments_only", "purchase_with_downpayment"},
        "new_build_mode": {"schedule", "base_seasonal", "annual_total"},
        "asset_expiry_mode": {"release", "retain", "salvage"},
    }
    for key, options in enum_options.items():
        if safe.get(key) not in options:
            safe[key] = enum_defaults[key]
            warnings.append(f"{key} was invalid and reset to {enum_defaults[key]}.")

    # Clamp any negative numeric values to zero.
    for key, val in list(safe.items()):
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)) and val < 0:
            safe[key] = 0 if isinstance(val, int) else 0.0
            warnings.append(f"{key} was negative and was reset to 0.")

    return safe, warnings


def _json_safe_value(value):
    if value is None:
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _json_default(value):
    return _json_safe_value(value)


def _serialize_dataframe_for_ai(df: pd.DataFrame) -> dict:
    if not isinstance(df, pd.DataFrame):
        return {"columns": [], "dtypes": {}, "row_count": 0, "records": []}
    safe_df = df.copy()
    for col in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
            safe_df[col] = pd.to_datetime(safe_df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    records = []
    for row in safe_df.to_dict(orient="records"):
        records.append({str(k): _json_safe_value(v) for k, v in row.items()})
    return {
        "columns": [str(c) for c in safe_df.columns],
        "dtypes": {str(c): str(safe_df[c].dtype) for c in safe_df.columns},
        "row_count": int(len(safe_df)),
        "records": records,
    }


def _serialize_metrics_for_ai(metrics: dict) -> dict:
    out: dict = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, pd.DataFrame):
            out[key] = _serialize_dataframe_for_ai(value)
        elif isinstance(value, pd.Series):
            out[key] = [{"index": _json_safe_value(i), "value": _json_safe_value(v)} for i, v in value.items()]
        elif isinstance(value, dict):
            out[key] = {str(k): _json_safe_value(v) for k, v in value.items()}
        else:
            out[key] = _json_safe_value(value)
    return out


def _attach_metric_attrs_for_ai(df: pd.DataFrame, assumptions: dict) -> None:
    df.attrs["attach_rate"] = assumptions.get("attach_rate", 0.0)
    df.attrs["ar_days"] = assumptions.get("ar_days", 0.0)
    df.attrs["ap_days"] = assumptions.get("ap_days", 0.0)
    df.attrs["inventory_days"] = assumptions.get("inventory_days", 0.0)


def _resolve_ai_range_labels(value_df: pd.DataFrame, ui_state: dict | None, explicit_range: tuple[str, str] | None) -> tuple[str, str]:
    labels = value_df["Year_Month_Label"].tolist()
    if not labels:
        return "", ""
    if explicit_range and len(explicit_range) == 2:
        start_label = str(explicit_range[0])
        end_label = str(explicit_range[1])
    elif isinstance(ui_state, dict):
        start_label = str(ui_state.get("range_start_label", ""))
        end_label = str(ui_state.get("range_end_label", ""))
    else:
        start_label = ""
        end_label = ""
    if start_label not in labels:
        start_label = labels[0]
    if end_label not in labels:
        end_label = labels[-1]
    return start_label, end_label


def _build_ai_scenario_payload(
    *,
    scenario_name: str,
    source_type: str,
    raw_assumptions: dict,
    ui_state: dict | None = None,
    explicit_range: tuple[str, str] | None = None,
    import_warnings: list[str] | None = None,
    import_unknown_keys: list[str] | None = None,
) -> dict:
    assumptions, schema_warnings, unknown_keys = migrate_assumptions(raw_assumptions if isinstance(raw_assumptions, dict) else {})
    assumptions, normalization_warnings = _normalize_inputs_for_runtime(assumptions)
    assumptions_json = _serialize_assumptions(assumptions)
    nominal_df = _run_model_cached(assumptions_json)
    _attach_metric_attrs_for_ai(nominal_df, assumptions)

    value_mode = str(assumptions.get("value_mode", "nominal"))
    value_df = apply_value_mode(nominal_df, assumptions, value_mode)
    _attach_metric_attrs_for_ai(value_df, assumptions)

    range_start_label, range_end_label = _resolve_ai_range_labels(value_df, ui_state, explicit_range)
    range_df = _filter_df(value_df, range_start_label, range_end_label) if range_start_label and range_end_label else value_df

    full_horizon_months = int(assumptions.get("horizon_months", len(value_df)))
    metrics_full = compute_metrics(value_df, full_horizon_months)
    metrics_selected_range = compute_metrics(range_df, len(range_df))
    annual_kpis_full = _build_annual_kpis(value_df, metrics_full)
    annual_kpis_selected_range = _build_annual_kpis(range_df, metrics_selected_range)
    integrity_findings = run_integrity_checks(nominal_df, assumptions, tol=1e-3)
    input_ts_df = _build_input_timeseries(assumptions, pd.DatetimeIndex(nominal_df["Date"]))

    all_warnings = list(
        dict.fromkeys(
            [
                *(import_warnings or []),
                *(schema_warnings or []),
                *(normalization_warnings or []),
            ]
        )
    )
    all_unknown_keys = sorted(set((import_unknown_keys or []) + (unknown_keys or [])))

    return {
        "scenario_name": scenario_name,
        "source_type": source_type,
        "assumptions_raw_input": _json_safe_value(raw_assumptions if isinstance(raw_assumptions, dict) else {}),
        "assumptions_runtime": _json_safe_value(assumptions),
        "ui_state": _json_safe_value(ui_state or {}),
        "value_mode": value_mode,
        "range_context": {
            "range_start_label": range_start_label,
            "range_end_label": range_end_label,
            "selected_range_months": int(len(range_df)),
        },
        "warnings": all_warnings,
        "unknown_keys": all_unknown_keys,
        "outputs": {
            "nominal_monthly": _serialize_dataframe_for_ai(nominal_df),
            "value_mode_monthly": _serialize_dataframe_for_ai(value_df),
            "selected_range_value_mode_monthly": _serialize_dataframe_for_ai(range_df),
            "input_time_series_monthly": _serialize_dataframe_for_ai(input_ts_df),
            "annual_kpis_full_horizon": _serialize_dataframe_for_ai(annual_kpis_full),
            "annual_kpis_selected_range": _serialize_dataframe_for_ai(annual_kpis_selected_range),
            "metrics_full_horizon": _serialize_metrics_for_ai(metrics_full),
            "metrics_selected_range": _serialize_metrics_for_ai(metrics_selected_range),
            "integrity_findings": _json_safe_value(integrity_findings),
        },
    }


def _ai_transformation_logic_summary() -> dict:
    return {
        "summary": "Core transformation identities and calculation flow used in the HVAC cashflow model.",
        "pipeline_steps": [
            "Migrate and sanitize assumptions to schema v2.",
            "Validate inputs and enforce numeric/domain constraints.",
            "Build monthly staffing, operational drivers, and segment-level revenue streams.",
            "Compute direct costs, OPEX, EBITDA, working capital, capex, financing, and cash balances.",
            "Apply value-mode transformations for display (nominal, real inflation-adjusted, real present value).",
            "Compute KPI summaries and run accounting integrity checks.",
        ],
        "core_identities": [
            {"name": "Total Revenue", "formula": "Service + Replacement + Maintenance + Upsell + New Build"},
            {
                "name": "Total Direct Costs",
                "formula": "Service Materials + Replacement Equipment + Permits + Disposal + Direct Labor + Maintenance Direct Cost + Financing Fee Cost + Upsell Direct Cost + New Build Direct Cost",
            },
            {"name": "Gross Profit", "formula": "Total Revenue - Total Direct Costs"},
            {"name": "Total OPEX", "formula": "Fixed OPEX + Marketing Spend + Fleet Cost + Sales Payroll + Management Payroll"},
            {"name": "EBITDA", "formula": "Gross Profit - Total OPEX"},
            {"name": "NWC", "formula": "AR Balance + Inventory Balance - AP Balance"},
            {"name": "Change in NWC", "formula": "NWC[t] - NWC[t-1] (with opening NWC baseline)"},
            {"name": "Operating Cash Flow", "formula": "EBITDA - Change in NWC"},
            {"name": "Gross Capex", "formula": "Tools Capex + Truck Capex"},
            {"name": "Capex (Net)", "formula": "Gross Capex - Asset Salvage Proceeds"},
            {"name": "Free Cash Flow", "formula": "Operating Cash Flow - Capex"},
            {"name": "Net Financing Cash Flow", "formula": "-Term Loan Payment - LOC Interest + LOC Draw - LOC Repay - Owner Distributions"},
            {"name": "Net Cash Flow", "formula": "Operating Cash Flow - Capex + Net Financing Cash Flow"},
            {"name": "End Cash", "formula": "Begin Cash + Net Cash Flow"},
        ],
        "value_mode_logic": [
            {"mode": "nominal", "formula": "No transformation"},
            {"mode": "real_inflation", "formula": "Monetary columns / (1 + monthly_cost_inflation)^t"},
            {"mode": "real_pv", "formula": "Monetary columns / (1 + annual_discount_rate)^(t/12)"},
        ],
        "implementation_references": [
            {"file": "src/schema.py", "function": "migrate_assumptions"},
            {"file": "src/model.py", "function": "run_model"},
            {"file": "src/value_modes.py", "function": "apply_value_mode"},
            {"file": "src/metrics.py", "function": "compute_metrics"},
            {"file": "src/integrity_checks.py", "function": "run_integrity_checks"},
            {"file": "app.py", "function": "_normalize_inputs_for_runtime"},
            {"file": "app.py", "function": "_build_input_timeseries"},
        ],
    }


def _load_ai_source_snapshots(include_source_code: bool) -> dict[str, str]:
    if not include_source_code:
        return {}
    source_code: dict[str, str] = {}
    for rel_path in AI_EXPORT_SOURCE_FILES:
        p = Path(rel_path)
        if not p.exists():
            continue
        try:
            source_code[rel_path] = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source_code[rel_path] = p.read_text(encoding="utf-8", errors="replace")
    return source_code


def _build_ai_export_pack(
    *,
    scope: str,
    active_assumptions: dict,
    active_ui_state: dict,
    active_range: tuple[str, str],
    selected_scenarios: list[str],
    selected_workspaces: list[str],
    include_source_code: bool,
    include_runtime_logs: bool,
    runtime_log_limit: int,
    active_input_warnings: list[str],
    active_integrity_findings: list[dict],
    active_goal_seek_result: dict | None,
) -> dict:
    scenarios: list[dict] = []
    failures: list[dict] = []
    requested_scenarios = selected_scenarios or []
    requested_workspaces = selected_workspaces or []

    def _append_failure(source_type: str, name: str, message: str) -> None:
        failures.append({"source_type": source_type, "name": name, "message": message})

    include_active = scope in {"Active scenario only", "Active + selected saved items"}
    include_saved_scenarios = scope in {"Saved scenarios (select multiple)", "Active + selected saved items"}
    include_saved_workspaces = scope in {"Saved workspaces (select multiple)", "Active + selected saved items"}

    if include_active:
        try:
            active_payload = _build_ai_scenario_payload(
                scenario_name="active_scenario",
                source_type="active",
                raw_assumptions=active_assumptions,
                ui_state=active_ui_state,
                explicit_range=active_range,
                import_warnings=active_input_warnings,
            )
            active_payload["active_runtime_context"] = {
                "input_warnings": _json_safe_value(active_input_warnings),
                "integrity_findings": _json_safe_value(active_integrity_findings),
                "goal_seek_result": _json_safe_value(active_goal_seek_result or {}),
            }
            scenarios.append(active_payload)
        except Exception as exc:
            _append_failure("active", "active_scenario", str(exc))

    if include_saved_scenarios:
        for scenario_name in requested_scenarios:
            bundle = load_saved(SCENARIO_TYPE, scenario_name)
            if bundle is None:
                _append_failure("saved_scenario", scenario_name, "Saved scenario not found.")
                continue
            assumptions, ui_state, import_warnings, import_unknown = parse_import_json(json.dumps(bundle))
            try:
                scenarios.append(
                    _build_ai_scenario_payload(
                        scenario_name=scenario_name,
                        source_type="saved_scenario",
                        raw_assumptions=assumptions,
                        ui_state=ui_state,
                        import_warnings=import_warnings,
                        import_unknown_keys=import_unknown,
                    )
                )
            except Exception as exc:
                _append_failure("saved_scenario", scenario_name, str(exc))

    if include_saved_workspaces:
        for workspace_name in requested_workspaces:
            bundle = load_saved(WORKSPACE_TYPE, workspace_name)
            if bundle is None:
                _append_failure("saved_workspace", workspace_name, "Saved workspace not found.")
                continue
            assumptions, ui_state, import_warnings, import_unknown = parse_import_json(json.dumps(bundle))
            try:
                scenarios.append(
                    _build_ai_scenario_payload(
                        scenario_name=workspace_name,
                        source_type="saved_workspace",
                        raw_assumptions=assumptions,
                        ui_state=ui_state,
                        import_warnings=import_warnings,
                        import_unknown_keys=import_unknown,
                    )
                )
            except Exception as exc:
                _append_failure("saved_workspace", workspace_name, str(exc))

    runtime_log_rows = read_runtime_events(limit=max(0, int(runtime_log_limit))) if include_runtime_logs else []
    generated_at = pd.Timestamp.utcnow().isoformat()
    return {
        "export_type": "hvac_ai_context_pack",
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "selection": {
            "scope": scope,
            "selected_saved_scenarios": requested_scenarios,
            "selected_saved_workspaces": requested_workspaces,
        },
        "summary": {
            "scenario_count": int(len(scenarios)),
            "failure_count": int(len(failures)),
            "include_source_code": bool(include_source_code),
            "include_runtime_logs": bool(include_runtime_logs),
            "runtime_log_rows": int(len(runtime_log_rows)),
        },
        "failures": failures,
        "transformation_logic": _ai_transformation_logic_summary(),
        "source_code_snapshot": _load_ai_source_snapshots(include_source_code),
        "runtime_log_tail": _json_safe_value(runtime_log_rows),
        "scenarios": scenarios,
    }


def _template_value_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return isinstance(value, str) and value.strip() == ""


def _coerce_template_scalar(value, default_value):
    if _template_value_missing(value):
        return deepcopy(default_value)
    if isinstance(default_value, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        txt = str(value).strip().lower()
        if txt in {"1", "true", "yes", "y", "on"}:
            return True
        if txt in {"0", "false", "no", "n", "off"}:
            return False
        return deepcopy(default_value)
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        try:
            return int(float(str(value).replace(",", "")))
        except (TypeError, ValueError):
            return deepcopy(default_value)
    if isinstance(default_value, float):
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return deepcopy(default_value)
    if isinstance(default_value, str):
        return str(value).strip()
    return value


def _parse_template_json_list(value, *, field_name: str, warnings: list[str], scenario_name: str) -> list[dict]:
    if _template_value_missing(value):
        return []
    if isinstance(value, list):
        return value
    txt = str(value).strip()
    if not txt:
        return []
    try:
        parsed = json.loads(txt)
    except json.JSONDecodeError:
        warnings.append(f"{scenario_name}: `{field_name}` JSON could not be parsed and was ignored.")
        return []
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        warnings.append(f"{scenario_name}: `{field_name}` JSON must be a list and was ignored.")
        return []
    return parsed


def _template_sheet_lookup(sheet_names: list[str], target_sheet: str) -> str | None:
    target = str(target_sheet).strip().lower()
    for sheet in sheet_names:
        if str(sheet).strip().lower() == target:
            return sheet
    return None


def _coerce_template_month(value) -> str:
    if _template_value_missing(value):
        return ""
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).strftime("%Y-%m")
    txt = str(value).strip()
    if len(txt) >= 10 and txt[4] == "-" and txt[7] == "-":
        return txt[:7]
    return txt


def _parse_template_detail_sheet(key: str, df: pd.DataFrame, warnings: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    spec = SCENARIO_TEMPLATE_DETAIL_SPECS[key]
    required_cols = set(spec["columns"])
    existing_cols = {str(c).strip() for c in df.columns}
    missing_cols = sorted(required_cols - existing_cols)
    if missing_cols:
        warnings.append(f"Sheet `{spec['sheet']}` is missing required columns: {', '.join(missing_cols)}")
        return out

    normalized = df.rename(columns={c: str(c).strip() for c in df.columns})
    for idx, row in normalized.iterrows():
        scenario_name = str(row.get("scenario_name", "")).strip()
        if not scenario_name:
            continue
        month = _coerce_template_month(row.get("month"))
        if not month:
            warnings.append(f"{scenario_name}: row {idx + 2} in `{spec['sheet']}` missing month and was ignored.")
            continue
        if key in {"tech_staffing_events", "sales_staffing_events"}:
            try:
                hires = max(0, int(float(row.get("hires", 0) if not _template_value_missing(row.get("hires")) else 0)))
                attrition = max(
                    0,
                    int(float(row.get("attrition", 0) if not _template_value_missing(row.get("attrition")) else 0)),
                )
            except (TypeError, ValueError):
                warnings.append(f"{scenario_name}: row {idx + 2} in `{spec['sheet']}` has invalid hires/attrition.")
                continue
            out.setdefault(scenario_name, []).append({"month": month, "hires": hires, "attrition": attrition})
        else:
            try:
                installs = max(
                    0.0,
                    float(row.get("installs", 0.0) if not _template_value_missing(row.get("installs")) else 0.0),
                )
            except (TypeError, ValueError):
                warnings.append(f"{scenario_name}: row {idx + 2} in `{spec['sheet']}` has invalid installs.")
                continue
            out.setdefault(scenario_name, []).append({"month": month, "installs": installs})
    return out


def _parse_scenario_template_upload(uploaded_file) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    if uploaded_file is None:
        return [], ["No template file selected."]

    file_name = str(getattr(uploaded_file, "name", "template")).lower()
    raw_bytes = uploaded_file.getvalue()
    scenarios_df: pd.DataFrame | None = None
    detail_frames: dict[str, pd.DataFrame] = {}

    try:
        if file_name.endswith(".csv"):
            scenarios_df = pd.read_csv(BytesIO(raw_bytes), dtype=object)
        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            workbook = pd.ExcelFile(BytesIO(raw_bytes))
            scenario_sheet = _template_sheet_lookup(workbook.sheet_names, SCENARIO_TEMPLATE_SHEET_SCENARIOS) or workbook.sheet_names[0]
            scenarios_df = pd.read_excel(workbook, sheet_name=scenario_sheet, dtype=object)
            for key, spec in SCENARIO_TEMPLATE_DETAIL_SPECS.items():
                sheet_name = _template_sheet_lookup(workbook.sheet_names, spec["sheet"])
                if sheet_name is not None:
                    detail_frames[key] = pd.read_excel(workbook, sheet_name=sheet_name, dtype=object)
        else:
            return [], ["Template import supports `.csv`, `.xlsx`, or `.xls` files."]
    except Exception as exc:
        return [], [f"Template file could not be parsed: {exc}"]

    if scenarios_df is None or len(scenarios_df) == 0:
        return [], ["Template does not contain any scenario rows."]

    scenarios_df = scenarios_df.rename(columns={c: str(c).strip() for c in scenarios_df.columns})
    if "scenario_name" not in scenarios_df.columns:
        warnings.append("Template is missing `scenario_name`; fallback names were generated.")
        scenarios_df["scenario_name"] = [f"Scenario_{idx + 1}" for idx in range(len(scenarios_df))]

    scalar_keys = [k for k in DEFAULTS.keys() if k not in SCENARIO_TEMPLATE_COMPLEX_KEYS]
    known_cols = {"scenario_name", "notes"} | set(scalar_keys) | {f"{k}_json" for k in SCENARIO_TEMPLATE_COMPLEX_KEYS}
    unknown_cols = [c for c in scenarios_df.columns if c not in known_cols]
    if unknown_cols:
        warnings.append(f"Unknown template columns were ignored: {', '.join(unknown_cols)}")

    scenario_map: dict[str, dict] = {}
    scenario_order: list[str] = []
    for idx, row in scenarios_df.iterrows():
        scenario_name = str(row.get("scenario_name", "")).strip() or f"Scenario_{idx + 1}"
        if scenario_name in scenario_map:
            warnings.append(f"Duplicate scenario_name `{scenario_name}` detected; later row replaced prior row.")
        assumptions = deepcopy(DEFAULTS)
        for key in scalar_keys:
            if key in scenarios_df.columns and not _template_value_missing(row.get(key)):
                assumptions[key] = _coerce_template_scalar(row.get(key), DEFAULTS[key])
        for key in SCENARIO_TEMPLATE_COMPLEX_KEYS:
            json_col = f"{key}_json"
            if json_col in scenarios_df.columns and not _template_value_missing(row.get(json_col)):
                assumptions[key] = _parse_template_json_list(
                    row.get(json_col),
                    field_name=json_col,
                    warnings=warnings,
                    scenario_name=scenario_name,
                )
        scenario_map[scenario_name] = assumptions
        if scenario_name not in scenario_order:
            scenario_order.append(scenario_name)

    for key, df in detail_frames.items():
        parsed_rows_by_scenario = _parse_template_detail_sheet(key, df, warnings)
        for scenario_name, rows in parsed_rows_by_scenario.items():
            if scenario_name not in scenario_map:
                scenario_map[scenario_name] = deepcopy(DEFAULTS)
                scenario_order.append(scenario_name)
            if key in {"tech_staffing_events", "sales_staffing_events"}:
                scenario_map[scenario_name][key] = _coalesce_events_by_month(rows)
            else:
                scenario_map[scenario_name][key] = _coalesce_schedule_by_month(rows)

    parsed: list[dict] = []
    for scenario_name in scenario_order:
        migrated, migrate_warnings, _ = migrate_assumptions(scenario_map[scenario_name])
        parsed.append(
            {
                "scenario_name": scenario_name,
                "assumptions": migrated,
                "warnings": migrate_warnings,
            }
        )
        if migrate_warnings:
            warnings.append(f"{scenario_name}: {' | '.join(migrate_warnings[:3])}")

    return parsed, warnings


def _build_template_seed_scenarios(
    active_assumptions: dict,
    include_active: bool,
    selected_saved_scenarios: list[str],
) -> tuple[list[dict], list[str]]:
    seeds: list[dict] = []
    warnings: list[str] = []
    if include_active:
        seeds.append({"scenario_name": "active_scenario", "assumptions": deepcopy(active_assumptions)})
    for name in selected_saved_scenarios:
        bundle = load_saved(SCENARIO_TYPE, name)
        if bundle is None:
            warnings.append(f"Saved scenario `{name}` was not found and was skipped.")
            continue
        assumptions, _, import_warnings, _ = parse_import_json(json.dumps(bundle))
        seeds.append({"scenario_name": name, "assumptions": assumptions})
        if import_warnings:
            warnings.append(f"{name}: {' | '.join(import_warnings[:2])}")
    if not seeds:
        seeds.append({"scenario_name": "Template_1", "assumptions": deepcopy(DEFAULTS)})
    return seeds, warnings


def _default_type_label(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    return "str"


def _build_scenario_template_frames(seed_scenarios: list[dict]) -> dict[str, pd.DataFrame]:
    scalar_keys = [k for k in DEFAULTS.keys() if k not in SCENARIO_TEMPLATE_COMPLEX_KEYS]
    scenario_cols = ["scenario_name", "notes"] + scalar_keys + [f"{k}_json" for k in SCENARIO_TEMPLATE_COMPLEX_KEYS]
    scenario_rows = []
    detail_rows: dict[str, list[dict]] = {k: [] for k in SCENARIO_TEMPLATE_COMPLEX_KEYS}

    for idx, item in enumerate(seed_scenarios, start=1):
        scenario_name = str(item.get("scenario_name", "")).strip() or f"Scenario_{idx}"
        assumptions = item.get("assumptions", {}) if isinstance(item.get("assumptions"), dict) else {}
        row = {"scenario_name": scenario_name, "notes": ""}
        for key in scalar_keys:
            row[key] = assumptions.get(key, DEFAULTS[key])
        for key in SCENARIO_TEMPLATE_COMPLEX_KEYS:
            rows = assumptions.get(key, [])
            if key in {"tech_staffing_events", "sales_staffing_events"}:
                rows = _coalesce_events_by_month(rows if isinstance(rows, list) else [])
            else:
                rows = _coalesce_schedule_by_month(rows if isinstance(rows, list) else [])
            row[f"{key}_json"] = json.dumps(rows, ensure_ascii=False)
            for detail_row in rows:
                if key in {"tech_staffing_events", "sales_staffing_events"}:
                    detail_rows[key].append(
                        {
                            "scenario_name": scenario_name,
                            "month": detail_row.get("month", ""),
                            "hires": detail_row.get("hires", 0),
                            "attrition": detail_row.get("attrition", 0),
                        }
                    )
                else:
                    detail_rows[key].append(
                        {
                            "scenario_name": scenario_name,
                            "month": detail_row.get("month", ""),
                            "installs": detail_row.get("installs", 0.0),
                        }
                    )
        scenario_rows.append(row)

    readme_df = pd.DataFrame(
        [
            {
                "Step": "1",
                "Instruction": "Edit the `scenarios` sheet (one row per scenario). Keep `scenario_name` unique.",
            },
            {
                "Step": "2",
                "Instruction": "For staffing/new-build schedules, either edit the detail sheets or use the *_json columns on `scenarios`.",
            },
            {
                "Step": "3",
                "Instruction": "Save as `.xlsx` (preferred for multi-sheet) or `.csv` (scenarios sheet only).",
            },
            {
                "Step": "4",
                "Instruction": "Upload in Import/Export -> Scenario Template Import and choose apply/save actions.",
            },
            {
                "Step": "Note",
                "Instruction": "Boolean examples: TRUE/FALSE, 1/0, yes/no. Percents should be decimals (0.25 = 25%).",
            },
        ]
    )

    reference_rows = []
    for key, default_val in DEFAULTS.items():
        reference_rows.append(
            {
                "input_key": key,
                "default_value": json.dumps(default_val, ensure_ascii=False)
                if isinstance(default_val, (dict, list))
                else default_val,
                "expected_type": _default_type_label(default_val),
                "allowed_values": ", ".join(SCENARIO_TEMPLATE_ENUM_OPTIONS.get(key, [])),
                "guidance": str(INPUT_GUIDANCE.get(key, {}).get("note", "")),
            }
        )
    reference_df = pd.DataFrame(reference_rows)

    frames: dict[str, pd.DataFrame] = {
        SCENARIO_TEMPLATE_SHEET_README: readme_df,
        SCENARIO_TEMPLATE_SHEET_SCENARIOS: pd.DataFrame(scenario_rows, columns=scenario_cols),
        SCENARIO_TEMPLATE_SHEET_REFERENCE: reference_df,
    }
    for key, spec in SCENARIO_TEMPLATE_DETAIL_SPECS.items():
        frames[spec["sheet"]] = pd.DataFrame(detail_rows[key], columns=spec["columns"])
    return frames


def _build_scenario_template_excel_bytes(frames: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output.seek(0)
    return output.getvalue()


st.set_page_config(page_title="HVAC Cash Flow Model v2", layout="wide")
st.title("HVAC Cash Flow Model v2")
st.caption("Schema v2 model with staffing events, segmented maintenance and upsell, goal seek, and workspace persistence.")
st.markdown(
    """
    <style>
    :root {
        --sidebar-width: 620px;
    }
    section[data-testid="stSidebar"] {
        transition: width 0.2s ease, min-width 0.2s ease;
    }
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: min(var(--sidebar-width), 85vw) !important;
        min-width: min(var(--sidebar-width), 85vw) !important;
        max-width: 92vw !important;
        resize: horizontal;
        overflow: auto;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 2.5rem !important;
        min-width: 2.5rem !important;
    }
    @media (max-width: 1200px) {
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: min(var(--sidebar-width), 92vw) !important;
            min-width: min(var(--sidebar-width), 92vw) !important;
        }
    }
    section[data-testid="stSidebar"][aria-expanded="true"]::-webkit-resizer {
        background: rgba(60, 130, 246, 0.35);
    }
    .main .block-container {
        max-width: 100%;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }
    section[data-testid="stSidebar"] button[kind="secondary"] {
        width: 100%;
    }
    section[data-testid="stSidebar"] button[kind="primary"] {
        width: 100%;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid rgba(120, 120, 120, 0.25);
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.25rem;
    }
    div[data-baseweb="input"] input {
        font-variant-numeric: tabular-nums;
    }
    div[data-baseweb="input"] input[type="number"] {
        text-align: right;
    }
    div[data-testid="stMetricValue"] {
        font-variant-numeric: tabular-nums;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, deepcopy(v))
for k, v in UI_DEFAULTS.items():
    st.session_state.setdefault(k, deepcopy(v))
st.session_state.setdefault("_deferred_session_state_updates", {})
st.session_state.setdefault("runtime_log_limit", 120)
st.session_state.setdefault("_input_warning_log_signature", "")
st.session_state.setdefault("_integrity_log_signature", "")
st.session_state.setdefault("ai_export_scope", AI_EXPORT_SCOPE_OPTIONS[0])
st.session_state.setdefault("ai_export_selected_scenarios", [])
st.session_state.setdefault("ai_export_selected_workspaces", [])
st.session_state.setdefault("ai_export_include_source_code", True)
st.session_state.setdefault("ai_export_include_runtime_logs", True)
st.session_state.setdefault("ai_export_log_limit", 200)
st.session_state.setdefault("ai_export_payload_json", "")
st.session_state.setdefault("ai_export_filename", "")
st.session_state.setdefault("ai_export_summary", "")
st.session_state.setdefault("template_include_active", True)
st.session_state.setdefault("template_seed_saved_scenarios", [])
st.session_state.setdefault("template_import_overwrite", False)
_apply_deferred_state_updates()
if "preset_scenarios" not in st.session_state or not isinstance(st.session_state.get("preset_scenarios"), dict):
    st.session_state["preset_scenarios"] = deepcopy(PRESET_SCENARIOS)
for preset_name, preset_values in PRESET_SCENARIOS.items():
    st.session_state["preset_scenarios"].setdefault(preset_name, deepcopy(preset_values))
if not st.session_state.get("applied_assumptions_json"):
    st.session_state["applied_assumptions_json"] = _serialize_assumptions(_assumptions_from_state())

for slot in range(1, 5):
    st.session_state.setdefault(f"chart_{slot}_enabled", False)
    st.session_state.setdefault(f"chart_{slot}_type", "line")
    st.session_state.setdefault(f"chart_{slot}_cols", [])
    st.session_state.setdefault(f"chart_{slot}_title", f"Custom Chart {slot}")

manual_apply_inputs = False

with st.sidebar:
    st.caption("Tip: drag the lower-right corner of this sidebar to resize width.")
    st.header("Scenario Manager")
    preset_options = list(st.session_state["preset_scenarios"].keys())
    if not preset_options:
        st.session_state["preset_scenarios"] = deepcopy(PRESET_SCENARIOS)
        preset_options = list(st.session_state["preset_scenarios"].keys())
    if st.session_state.get("preset_choice") not in preset_options:
        st.session_state["preset_choice"] = preset_options[0]
    st.selectbox(
        "Preset",
        options=preset_options,
        key="preset_choice",
        help="Choose a preset scenario profile to apply to the model inputs.",
    )
    c1, c2 = st.columns(2)
    apply_preset = c1.button("Apply Preset", help="Apply the selected preset values to all model assumptions.")
    reset_base = c2.button("Reset Base", help="Reset current assumptions to the default base values.")

    if apply_preset:
        _apply_assumptions_to_state(st.session_state["preset_scenarios"][st.session_state["preset_choice"]])
        st.success(f"Applied {st.session_state['preset_choice']} preset.")
    if reset_base:
        _apply_assumptions_to_state(DEFAULTS)
        st.success("Reset to base defaults.")

    st.subheader("Run Controls")
    st.toggle(
        "Live Recalculate",
        key="auto_run_model",
        help="When off, edits are queued until you click Apply Input Changes.",
    )
    pending_keys = _pending_change_keys_from_state()
    pending_count = len(pending_keys)
    if pending_count:
        if st.session_state.get("auto_run_model", True):
            st.caption(f"Detected {pending_count} changes. Live mode will apply them on this rerun.")
        else:
            st.caption(f"Pending input changes: {pending_count}")
            with st.expander("Pending Fields", expanded=False):
                for key in pending_keys[:40]:
                    st.write(f"- `{key}`")
                if pending_count > 40:
                    st.caption(f"...and {pending_count - 40} more")
    manual_apply_inputs = st.button(
        "Apply Input Changes",
        type="primary",
        disabled=pending_count == 0 or st.session_state.get("auto_run_model", True),
        help="Run the model using your queued edits when Live Recalculate is turned off.",
    )
    if st.session_state.get("auto_run_model", True):
        st.caption("Turn off Live Recalculate to batch edits and apply once.")

    st.subheader("Local Save/Load")
    save_name_raw = st.text_input(
        "Save Name",
        value=st.session_state.get("active_workspace_name", ""),
        placeholder="e.g., 2026 Base Plan",
    )
    save_name = save_name_raw.strip()
    overwrite_save = st.checkbox("Overwrite if exists", value=False)
    st.toggle("Autosave active workspace", key="autosave_enabled")

    b1, b2 = st.columns(2)
    save_scenario_btn = b1.button(
        "Save Scenario",
        help="Save assumptions only as a reusable scenario file.",
        disabled=not bool(save_name),
    )
    save_workspace_btn = b2.button(
        "Save Workspace",
        help="Save assumptions plus UI state (charts, ranges, selections).",
        disabled=not bool(save_name),
    )
    if not save_name:
        st.caption("Enter a Save Name to enable save actions.")

    current_assumptions = _assumptions_from_state()
    if save_scenario_btn:
        ok, msg = save_named_bundle(
            SCENARIO_TYPE, save_name, build_scenario_bundle(save_name, current_assumptions), overwrite=overwrite_save
        )
        if ok:
            st.success("Scenario saved.")
        else:
            append_runtime_event(
                level="WARNING",
                event="save_scenario_failed",
                message=msg,
                context={"name": save_name, "overwrite": overwrite_save},
            )
            st.warning(msg)
    if save_workspace_btn:
        workspace_bundle = build_workspace_bundle(save_name, current_assumptions, _current_ui_state())
        ok, msg = save_named_bundle(WORKSPACE_TYPE, save_name, workspace_bundle, overwrite=overwrite_save)
        if ok:
            st.session_state["active_workspace_name"] = save_name
            st.success("Workspace saved.")
        else:
            append_runtime_event(
                level="WARNING",
                event="save_workspace_failed",
                message=msg,
                context={"name": save_name, "overwrite": overwrite_save},
            )
            st.warning(msg)
    scenario_names = list_saved_names(SCENARIO_TYPE)
    workspace_names = list_saved_names(WORKSPACE_TYPE)
    selected_saved_scenario = st.selectbox("Saved Scenarios", [""] + scenario_names)
    selected_saved_workspace = st.selectbox("Saved Workspaces", [""] + workspace_names)

    st.subheader("Preset Management")
    if st.session_state.get("preset_edit_slot") not in preset_options:
        st.session_state["preset_edit_slot"] = preset_options[0]
    st.selectbox(
        "Preset Slot",
        options=preset_options,
        key="preset_edit_slot",
        help="Choose which preset slot (Base/Upside/Downside) you want to overwrite.",
    )
    pm1, pm2 = st.columns(2)
    save_current_to_preset_btn = pm1.button(
        "Save Current to Slot",
        help="Overwrite the selected preset slot using the currently loaded assumptions.",
    )
    save_saved_to_preset_btn = pm2.button(
        "Use Saved Scenario",
        disabled=not selected_saved_scenario,
        help="Overwrite the selected preset slot using the selected saved scenario.",
    )
    restore_default_presets_btn = st.button(
        "Restore Default Presets",
        help="Reset Base/Upside/Downside preset definitions back to system defaults.",
    )

    if save_current_to_preset_btn:
        slot = st.session_state["preset_edit_slot"]
        st.session_state["preset_scenarios"][slot] = deepcopy(_assumptions_from_state())
        st.success(f"Updated preset slot: {slot}")

    if save_saved_to_preset_btn and selected_saved_scenario:
        bundle = load_saved(SCENARIO_TYPE, selected_saved_scenario)
        if bundle is None:
            append_runtime_event(
                level="WARNING",
                event="preset_source_missing",
                message="Saved scenario not found while assigning preset.",
                context={"scenario_name": selected_saved_scenario},
            )
            st.warning(f"Saved scenario `{selected_saved_scenario}` was not found.")
        else:
            preset_assumptions, _, preset_warnings, preset_unknown = parse_import_json(json.dumps(bundle))
            slot = st.session_state["preset_edit_slot"]
            st.session_state["preset_scenarios"][slot] = deepcopy(preset_assumptions)
            if preset_warnings:
                st.warning(" | ".join(preset_warnings))
            if preset_unknown:
                st.info(f"Ignored unknown keys while assigning preset: {', '.join(preset_unknown)}")
            st.success(f"Preset slot `{slot}` now uses saved scenario `{selected_saved_scenario}`.")

    if restore_default_presets_btn:
        st.session_state["preset_scenarios"] = deepcopy(PRESET_SCENARIOS)
        if st.session_state.get("preset_choice") not in st.session_state["preset_scenarios"]:
            st.session_state["preset_choice"] = list(st.session_state["preset_scenarios"].keys())[0]
        st.success("Restored default preset definitions.")

    l1, l2, l3, l4 = st.columns(4)
    load_scenario_btn = l1.button("Load Scenario", help="Load the selected saved scenario into current assumptions.")
    load_workspace_btn = l2.button("Load Workspace", help="Load the selected saved workspace (assumptions + UI state).")
    delete_scenario_btn = l3.button(
        "Delete Scenario",
        help="Delete only the selected saved scenario from local storage.",
        disabled=not bool(selected_saved_scenario),
    )
    delete_workspace_btn = l4.button(
        "Delete Workspace",
        help="Delete only the selected saved workspace from local storage.",
        disabled=not bool(selected_saved_workspace),
    )

    if (load_scenario_btn and not selected_saved_scenario) or (load_workspace_btn and not selected_saved_workspace):
        st.info("Select a saved scenario/workspace before loading.")

    if load_scenario_btn and selected_saved_scenario:
        bundle = load_saved(SCENARIO_TYPE, selected_saved_scenario)
        if bundle is None:
            append_runtime_event(
                level="WARNING",
                event="load_scenario_missing",
                message="Saved scenario not found.",
                context={"scenario_name": selected_saved_scenario},
            )
            st.warning(f"Saved scenario `{selected_saved_scenario}` was not found.")
        else:
            assumptions, _, warnings, unknown = parse_import_json(json.dumps(bundle))
            _apply_assumptions_to_state(assumptions)
            if warnings:
                st.warning(" | ".join(warnings))
            if unknown:
                st.info(f"Ignored unknown keys: {', '.join(unknown)}")
            st.success(f"Loaded scenario: {selected_saved_scenario}")

    if load_workspace_btn and selected_saved_workspace:
        bundle = load_saved(WORKSPACE_TYPE, selected_saved_workspace)
        if bundle is None:
            append_runtime_event(
                level="WARNING",
                event="load_workspace_missing",
                message="Saved workspace not found.",
                context={"workspace_name": selected_saved_workspace},
            )
            st.warning(f"Saved workspace `{selected_saved_workspace}` was not found.")
        else:
            assumptions, ui_state, warnings, unknown = parse_import_json(json.dumps(bundle))
            _apply_assumptions_to_state(assumptions)
            _apply_ui_state(ui_state or {})
            st.session_state["active_workspace_name"] = selected_saved_workspace
            if warnings:
                st.warning(" | ".join(warnings))
            if unknown:
                st.info(f"Ignored unknown keys: {', '.join(unknown)}")
            st.success(f"Loaded workspace: {selected_saved_workspace}")

    if delete_scenario_btn and selected_saved_scenario:
        if delete_saved(SCENARIO_TYPE, selected_saved_scenario):
            st.success(f"Deleted scenario: {selected_saved_scenario}")
    if delete_workspace_btn and selected_saved_workspace:
        if delete_saved(WORKSPACE_TYPE, selected_saved_workspace):
            st.success(f"Deleted workspace: {selected_saved_workspace}")

    st.subheader("Import/Export")
    import_file = st.file_uploader("Import Scenario/Workspace JSON", type=["json"])
    import_btn = st.button(
        "Apply Imported JSON",
        disabled=import_file is None,
        help="Replace current assumptions/workspace with the uploaded JSON bundle.",
    )
    if import_btn and import_file is not None:
        try:
            import_text = import_file.getvalue().decode("utf-8")
        except UnicodeDecodeError as exc:
            append_runtime_event(
                level="ERROR",
                event="import_decode_failed",
                message="Import failed: file is not valid UTF-8 JSON.",
                context={"file_name": getattr(import_file, "name", "unknown")},
                exc=exc,
            )
            st.error("Import failed: file is not valid UTF-8 JSON.")
        else:
            assumptions, ui_state, warnings, unknown = parse_import_json(import_text)
            _apply_assumptions_to_state(assumptions)
            _apply_ui_state(ui_state or {})
            if warnings:
                st.warning(" | ".join(warnings))
            if unknown:
                st.info(f"Ignored unknown keys: {', '.join(unknown)}")
            st.success("Imported assumptions/workspace applied.")

    scenario_bundle_json = json.dumps(build_scenario_bundle("scenario_export", _assumptions_from_state()), indent=2)
    workspace_bundle_json = json.dumps(
        build_workspace_bundle("workspace_export", _assumptions_from_state(), _current_ui_state()),
        indent=2,
    )
    st.download_button(
        "Export Scenario JSON",
        scenario_bundle_json,
        file_name="hvac_scenario_v2.json",
        mime="application/json",
        help="Download assumptions only as a scenario bundle.",
    )
    st.download_button(
        "Export Workspace JSON",
        workspace_bundle_json,
        file_name="hvac_workspace_v2.json",
        mime="application/json",
        help="Download assumptions and UI configuration as a workspace bundle.",
    )

    st.subheader("Scenario Template (CSV/Excel)")
    st.caption(
        "Export a user-friendly template for creating one or many scenarios (including AI-assisted generation) and upload it back."
    )
    st.checkbox(
        "Include active scenario as a seed row",
        key="template_include_active",
        help="When enabled, the current in-app assumptions are added as a row in the template.",
    )
    template_saved_scenario_names = list_saved_names(SCENARIO_TYPE)
    st.multiselect(
        "Seed template with saved scenarios",
        options=template_saved_scenario_names,
        key="template_seed_saved_scenarios",
        help="Optional: pre-populate additional rows from existing saved scenarios.",
    )
    template_seed_scenarios, template_seed_warnings = _build_template_seed_scenarios(
        active_assumptions=_assumptions_from_state(),
        include_active=bool(st.session_state["template_include_active"]),
        selected_saved_scenarios=st.session_state.get("template_seed_saved_scenarios", []),
    )
    template_frames = _build_scenario_template_frames(template_seed_scenarios)
    template_scenarios_df = template_frames[SCENARIO_TEMPLATE_SHEET_SCENARIOS]
    if template_seed_warnings:
        st.caption(" | ".join(template_seed_warnings))
    st.caption(
        f"Template currently includes {len(template_seed_scenarios)} scenario row(s). "
        "Use Excel for full multi-sheet detail tables."
    )
    with st.expander("Template Preview (Scenarios Sheet)", expanded=False):
        st.dataframe(_format_dataframe_for_display(template_scenarios_df), width="stretch", hide_index=True)

    template_excel_bytes = None
    try:
        template_excel_bytes = _build_scenario_template_excel_bytes(template_frames)
    except Exception as exc:
        append_runtime_event(
            level="ERROR",
            event="template_excel_build_failed",
            message="Scenario template Excel export failed.",
            context={"seed_count": len(template_seed_scenarios)},
            exc=exc,
        )
        st.warning("Excel template could not be generated in this environment. CSV exports are still available.")
    if template_excel_bytes is not None:
        st.download_button(
            "Download Scenario Template (Excel)",
            template_excel_bytes,
            file_name="hvac_scenario_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Best option for multi-scenario editing with detail sheets for events and schedules.",
        )
    st.download_button(
        "Download Scenario Template (CSV - Scenarios Sheet)",
        template_scenarios_df.to_csv(index=False),
        file_name="hvac_scenario_template.csv",
        mime="text/csv",
        help="CSV includes one row per scenario plus JSON columns for events/schedules.",
    )
    with st.expander("Download Detail CSV Sheets", expanded=False):
        for key, spec in SCENARIO_TEMPLATE_DETAIL_SPECS.items():
            detail_df = template_frames.get(spec["sheet"], pd.DataFrame(columns=spec["columns"]))
            st.download_button(
                f"Download {spec['sheet']}.csv",
                detail_df.to_csv(index=False),
                file_name=f"{spec['sheet']}.csv",
                mime="text/csv",
            )

    template_upload = st.file_uploader(
        "Import Scenario Template (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        key="template_upload_file",
        help="Upload the filled template to apply one scenario or save many scenarios at once.",
    )
    if template_upload is not None:
        parsed_template_scenarios, template_import_warnings = _parse_scenario_template_upload(template_upload)
        if template_import_warnings:
            with st.expander(f"Template Import Warnings ({len(template_import_warnings)})", expanded=False):
                for warning in template_import_warnings:
                    st.write(f"- {warning}")
        if parsed_template_scenarios:
            scenario_names = [item["scenario_name"] for item in parsed_template_scenarios]
            st.caption(f"Parsed {len(parsed_template_scenarios)} scenario(s): {', '.join(scenario_names[:8])}")
            preview_rows = []
            for item in parsed_template_scenarios:
                preview_rows.append(
                    {
                        "Scenario": item["scenario_name"],
                        "Start Year": item["assumptions"].get("start_year"),
                        "Start Month": item["assumptions"].get("start_month"),
                        "Horizon Months": item["assumptions"].get("horizon_months"),
                        "Warnings": len(item.get("warnings", [])),
                    }
                )
            with st.expander("Parsed Scenario Preview", expanded=False):
                st.dataframe(_format_dataframe_for_display(pd.DataFrame(preview_rows)), width="stretch", hide_index=True)

            st.checkbox(
                "Overwrite existing saved scenarios on template import",
                key="template_import_overwrite",
            )
            selected_template_apply = st.selectbox(
                "Template scenario to apply now",
                options=[""] + scenario_names,
                key="template_apply_scenario_name",
            )
            t_apply_col, t_save_col = st.columns(2)
            apply_template_scenario_btn = t_apply_col.button(
                "Apply Selected Template Scenario",
                disabled=not bool(selected_template_apply),
            )
            save_template_scenarios_btn = t_save_col.button("Save All Template Scenarios")

            if apply_template_scenario_btn and selected_template_apply:
                selected = next((item for item in parsed_template_scenarios if item["scenario_name"] == selected_template_apply), None)
                if selected is None:
                    st.warning("Selected scenario was not found in parsed template payload.")
                else:
                    _apply_assumptions_to_state(selected["assumptions"])
                    st.success(f"Applied template scenario: {selected_template_apply}")

            if save_template_scenarios_btn:
                saved_count = 0
                failed_messages: list[str] = []
                for item in parsed_template_scenarios:
                    scenario_name = str(item["scenario_name"]).strip()
                    if not scenario_name:
                        failed_messages.append("Encountered a template scenario with blank name.")
                        continue
                    ok, msg = save_named_bundle(
                        SCENARIO_TYPE,
                        scenario_name,
                        build_scenario_bundle(scenario_name, item["assumptions"]),
                        overwrite=bool(st.session_state.get("template_import_overwrite", False)),
                    )
                    if ok:
                        saved_count += 1
                    else:
                        failed_messages.append(f"{scenario_name}: {msg}")
                if failed_messages:
                    append_runtime_event(
                        level="WARNING",
                        event="template_import_partial_failure",
                        message="One or more scenarios failed to save from template import.",
                        context={"failures": failed_messages[:25]},
                    )
                if saved_count:
                    st.success(f"Saved {saved_count} scenario(s) from template.")
                if failed_messages:
                    st.warning(" | ".join(failed_messages[:8]))
        else:
            st.info("No scenario rows were parsed from the uploaded template.")

    st.subheader("Runtime Diagnostics")
    log_path = Path(runtime_log_path())
    st.caption(f"Runtime log file: `{log_path}`")
    st.number_input(
        "Recent runtime log rows",
        min_value=20,
        max_value=2000,
        step=20,
        key="runtime_log_limit",
        help="Use this log to diagnose user-reported errors after deployment.",
    )
    runtime_events = read_runtime_events(limit=int(st.session_state["runtime_log_limit"]))
    if runtime_events:
        runtime_df = pd.DataFrame(runtime_events)
        preferred_cols = [
            "timestamp_utc",
            "level",
            "event",
            "message",
            "exception_type",
            "exception_message",
            "context",
        ]
        runtime_cols = [c for c in preferred_cols if c in runtime_df.columns] + [
            c for c in runtime_df.columns if c not in preferred_cols
        ]
        st.dataframe(_format_dataframe_for_display(runtime_df[runtime_cols]), width="stretch", hide_index=True)
    else:
        st.caption("No runtime events logged yet.")
    if log_path.exists():
        st.download_button(
            "Download Runtime Log (JSONL)",
            log_path.read_text(encoding="utf-8"),
            file_name="hvac_runtime_events.jsonl",
            mime="application/x-ndjson",
            help="Share this log file for post-release debugging and patch support.",
        )
        clear_runtime_log = st.button("Clear Runtime Log")
        if clear_runtime_log:
            try:
                log_path.unlink()
            except OSError as exc:
                append_runtime_event(
                    level="ERROR",
                    event="runtime_log_clear_failed",
                    message="Failed to clear runtime log file.",
                    context={"path": str(log_path)},
                    exc=exc,
                )
                st.warning("Could not clear runtime log file.")
            else:
                st.success("Runtime log cleared.")
                st.rerun()

    st.subheader("Helper Utilities")
    st.caption("Quick actions to speed scenario setup.")
    st.slider(
        "Revenue Shift %",
        min_value=-0.5,
        max_value=0.5,
        step=0.01,
        key="helper_revenue_shift_pct",
        help="Applies a bulk multiplier to core revenue drivers when you click Apply Quick Shift.",
    )
    st.slider(
        "Cost Shift %",
        min_value=-0.5,
        max_value=0.5,
        step=0.01,
        key="helper_cost_shift_pct",
        help="Applies a bulk multiplier to core cost and payroll drivers when you click Apply Quick Shift.",
    )
    hu1, hu2, hu3 = st.columns(3)
    apply_quick_shift = hu1.button("Apply Quick Shift", help="Apply the Revenue Shift % and Cost Shift % multipliers to core drivers.")
    clear_events = hu2.button("Clear Staffing Events", help="Clear all tech and sales staffing event rows.")
    clear_nb_schedules = hu3.button("Clear New-Build Schedules", help="Clear residential and light-commercial new-build schedule rows.")
    if apply_quick_shift:
        _apply_helper_shifts(st.session_state["helper_revenue_shift_pct"], st.session_state["helper_cost_shift_pct"])
        _set_session_state_or_defer("helper_revenue_shift_pct", 0.0)
        _set_session_state_or_defer("helper_cost_shift_pct", 0.0)
        st.success("Quick revenue/cost shift applied.")
    if clear_events:
        st.session_state["tech_staffing_events"] = []
        st.session_state["sales_staffing_events"] = []
        st.session_state["tech_staffing_events_master"] = []
        st.session_state["sales_staffing_events_master"] = []
        st.session_state.pop("tech_staffing_events_editor", None)
        st.session_state.pop("sales_staffing_events_editor", None)
        st.success("Cleared tech and sales staffing events.")
    if clear_nb_schedules:
        st.session_state["res_new_build_install_schedule"] = []
        st.session_state["lc_new_build_install_schedule"] = []
        st.session_state["res_new_build_install_schedule_master"] = []
        st.session_state["lc_new_build_install_schedule_master"] = []
        st.session_state.pop("res_new_build_install_schedule_editor", None)
        st.session_state.pop("lc_new_build_install_schedule_editor", None)
        st.success("Cleared new-build install schedules.")

    st.subheader("Input Navigation")
    input_sections = [
        "All Sections",
        "Model Controls",
        "Staffing and Ops",
        "Service and Replacement",
        "Maintenance",
        "Upsell",
        "New-Build Installs",
        "Marketing and Fleet",
        "Overhead and Management",
        "Working Capital, Debt, and Distributions",
    ]
    st.selectbox(
        "Input Focus",
        options=input_sections,
        key="input_focus_section",
        help="Show one section at a time for faster editing.",
    )
    if st.session_state["input_focus_section"] != "All Sections":
        st.session_state["last_active_input_section"] = st.session_state["input_focus_section"]
    st.toggle(
        "Expand all sections",
        key="expand_all_input_sections",
        help="When Input Focus is set to All Sections, expand all input groups.",
    )
    if st.session_state["input_focus_section"] == "All Sections" and not st.session_state.get("expand_all_input_sections", False):
        st.caption("Accordion mode: the app keeps your most recently edited section open.")
    st.text_input(
        "Find Input",
        key="input_search_query",
        placeholder="Try: churn, wage, truck, discount",
        help="Search by input key or guidance text and jump to the matching section.",
    )
    input_matches = _search_input_keys(st.session_state.get("input_search_query", ""))
    if st.session_state.get("input_search_query"):
        if input_matches:
            match_options = [f"{key}  |  {section}" for key, section, _ in input_matches]
            if st.session_state.get("input_match_choice") not in match_options:
                st.session_state["input_match_choice"] = match_options[0]
            selected_match = st.selectbox("Matching Inputs", options=match_options, key="input_match_choice")
            match_idx = match_options.index(selected_match)
            match_key, match_section, match_note = input_matches[match_idx]
            if st.session_state.get("input_focus_section") != match_section:
                st.session_state["input_focus_section"] = match_section
                st.session_state["last_active_input_section"] = match_section
                st.caption(f"Focused section: {match_section}")
            if match_note:
                st.caption(f"`{match_key}`: {match_note}")
        else:
            st.caption("No matching inputs found.")
    else:
        st.session_state.pop("input_match_choice", None)

    def _show_section(name: str) -> bool:
        focus = st.session_state["input_focus_section"]
        return focus == "All Sections" or focus == name

    def _section_expanded(name: str) -> bool:
        focus = st.session_state["input_focus_section"]
        if focus == "All Sections":
            if bool(st.session_state.get("expand_all_input_sections", False)):
                return True
            active = st.session_state.get("last_active_input_section", "Model Controls")
            if active not in input_sections:
                active = "Model Controls"
                st.session_state["last_active_input_section"] = active
            return name == active
        st.session_state["last_active_input_section"] = focus
        return focus == name

    month_opts = [month_name_option(m) for m in range(1, 13)]

    if _show_section("Model Controls"):
        with st.expander("Model Controls", expanded=_section_expanded("Model Controls")):
            mc_left, mc_right = st.columns(2)
            with mc_left:
                st.number_input("Start Year", min_value=2000, max_value=2100, step=1, key="start_year")
                st.slider("Horizon Months", 12, 120, key="horizon_months")
                st.slider(
                    "Monthly Price Growth",
                    min_value=0.0,
                    max_value=0.02,
                    step=0.0005,
                    key="monthly_price_growth",
                    help=help_with_guidance("monthly_price_growth", "Monthly increase applied to revenue price drivers."),
                )
                st.slider("Seasonality Amplitude", 0.0, 0.4, step=0.005, key="seasonality_amplitude")
                st.selectbox(
                    "Value Mode",
                    ["nominal", "real_inflation", "real_pv"],
                    key="value_mode",
                    help="Nominal, inflation-adjusted real, or discounted PV real view for all outputs.",
                )
            with mc_right:
                start_month_opt = st.selectbox(
                    "Start Month",
                    month_opts,
                    index=int(st.session_state["start_month"]) - 1,
                    key="start_month_opt",
                )
                st.session_state["start_month"] = parse_month_name_option(start_month_opt)
                st.slider(
                    "Monthly Cost Inflation",
                    min_value=0.0,
                    max_value=0.02,
                    step=0.0005,
                    key="monthly_cost_inflation",
                    help=help_with_guidance("monthly_cost_inflation", "Monthly increase applied to cost drivers."),
                )
                peak_month_opt = st.selectbox(
                    "Peak Month",
                    month_opts,
                    index=int(st.session_state["peak_month"]) - 1,
                    key="peak_month_opt",
                )
                st.session_state["peak_month"] = parse_month_name_option(peak_month_opt)
                st.slider(
                    "Discount Rate (Annual Nominal)",
                    0.0,
                    0.5,
                    step=0.005,
                    key="discount_rate_annual_nominal",
                    help=help_with_guidance("discount_rate_annual_nominal", "Used for discounted PV real-dollar presentation."),
                )

    if _show_section("Staffing and Ops"):
        with st.expander("Staffing and Ops", expanded=_section_expanded("Staffing and Ops")):
            staff_left, staff_right = st.columns(2)
            with staff_left:
                st.number_input("Starting Techs", min_value=0, step=1, key="starting_techs")
                st.number_input("Starting Sales Staff", min_value=0, step=1, key="sales_starting_staff")
                st.number_input(
                    "Calls per Tech per Workday",
                    min_value=0.0,
                    key="calls_per_tech_per_day",
                    help=help_with_guidance("calls_per_tech_per_day", "Average service calls completed by each technician per workday."),
                )
                st.number_input(
                    "Tech Hours per Workday",
                    min_value=0.0,
                    key="tech_hours_per_day",
                    help=help_with_guidance("tech_hours_per_day", "Paid technician hours for each working day."),
                )
                st.number_input(
                    "Tech Wage per Hour",
                    min_value=0.0,
                    key="tech_wage_per_hour",
                    help=help_with_guidance("tech_wage_per_hour", "Hourly wage for technicians."),
                )
                st.number_input(
                    "Tools Capex per New Tech",
                    min_value=0.0,
                    key="tools_per_new_tech_capex",
                    help=help_with_guidance("tools_per_new_tech_capex", "Upfront tool investment per net new technician."),
                )
                st.slider("Residential Capacity %", 0.0, 1.0, step=0.01, key="res_capacity_pct")
            with staff_right:
                st.number_input("Max Techs", min_value=0, step=1, key="max_techs")
                st.number_input(
                    "Sales Hours per Workday",
                    min_value=0.0,
                    key="sales_hours_per_day",
                    help=help_with_guidance("sales_hours_per_day", "Paid sales hours for each working day."),
                )
                st.number_input(
                    "Sales Wage per Hour",
                    min_value=0.0,
                    key="sales_wage_per_hour",
                    help=help_with_guidance("sales_wage_per_hour", "Hourly wage for sales staff."),
                )
                st.slider(
                    "Payroll Burden %",
                    0.0,
                    1.0,
                    step=0.001,
                    key="payroll_burden_pct",
                    help=help_with_guidance("payroll_burden_pct", "Payroll taxes/benefits as a percent of wages."),
                )
                st.number_input(
                    "Asset Reuse Lag Months",
                    min_value=0,
                    step=1,
                    key="asset_reuse_lag_months",
                    help=help_with_guidance("asset_reuse_lag_months", "How long assets remain reusable after attrition."),
                )
                st.selectbox("Asset Expiry Mode", ["release", "retain", "salvage"], key="asset_expiry_mode")
                st.slider("Asset Salvage %", 0.0, 1.0, step=0.01, key="asset_salvage_pct")

            horizon_dates = pd.date_range(
                start=datetime(int(st.session_state["start_year"]), int(st.session_state["start_month"]), 1),
                periods=int(st.session_state["horizon_months"]),
                freq="MS",
            )
            month_labels = horizon_dates.strftime("%Y-%m").tolist()
            if st.session_state.get("tech_staffing_events_master") is None:
                st.session_state["tech_staffing_events_master"] = _sort_events(st.session_state.get("tech_staffing_events", []))
            if st.session_state.get("sales_staffing_events_master") is None:
                st.session_state["sales_staffing_events_master"] = _sort_events(st.session_state.get("sales_staffing_events", []))

            tech_in_horizon, tech_out_of_horizon = _partition_events_by_horizon(
                st.session_state.get("tech_staffing_events_master", []),
                month_labels,
            )
            sales_in_horizon, sales_out_of_horizon = _partition_events_by_horizon(
                st.session_state.get("sales_staffing_events_master", []),
                month_labels,
            )
            event_col_cfg = {
                "month": st.column_config.SelectboxColumn("Month", options=month_labels, required=True),
                "hires": st.column_config.NumberColumn("Hires", min_value=0, step=1),
                "attrition": st.column_config.NumberColumn("Attrition", min_value=0, step=1),
            }
            tech_col, sales_col = st.columns(2)
            with tech_col:
                st.caption("Technician staffing events (monthly hires and attrition)")
                tech_events_df = st.data_editor(
                    _events_df(tech_in_horizon),
                    num_rows="dynamic",
                    column_config=event_col_cfg,
                    key="tech_staffing_events_editor",
                    width="stretch",
                )
            with sales_col:
                st.caption("Sales staffing events (monthly hires and attrition)")
                sales_events_df = st.data_editor(
                    _events_df(sales_in_horizon),
                    num_rows="dynamic",
                    column_config=event_col_cfg,
                    key="sales_staffing_events_editor",
                    width="stretch",
                )
            edited_tech_in_horizon = _sort_events(_events_from_editor(tech_events_df))
            edited_sales_in_horizon = _sort_events(_events_from_editor(sales_events_df))
            st.session_state["tech_staffing_events_master"] = _merge_event_sets(edited_tech_in_horizon, tech_out_of_horizon)
            st.session_state["sales_staffing_events_master"] = _merge_event_sets(edited_sales_in_horizon, sales_out_of_horizon)
            st.session_state["tech_staffing_events"] = edited_tech_in_horizon
            st.session_state["sales_staffing_events"] = edited_sales_in_horizon

            if tech_out_of_horizon:
                st.info(
                    f"Preserved {len(tech_out_of_horizon)} tech staffing row(s) outside the current model horizon. "
                    "Increase horizon to view or edit them."
                )
            if sales_out_of_horizon:
                st.info(
                    f"Preserved {len(sales_out_of_horizon)} sales staffing row(s) outside the current model horizon. "
                    "Increase horizon to view or edit them."
                )
    else:
        horizon_dates = pd.date_range(
            start=datetime(int(st.session_state["start_year"]), int(st.session_state["start_month"]), 1),
            periods=int(st.session_state["horizon_months"]),
            freq="MS",
        )
        month_labels = horizon_dates.strftime("%Y-%m").tolist()
        if st.session_state.get("tech_staffing_events_master") is None:
            st.session_state["tech_staffing_events_master"] = _sort_events(st.session_state.get("tech_staffing_events", []))
        if st.session_state.get("sales_staffing_events_master") is None:
            st.session_state["sales_staffing_events_master"] = _sort_events(st.session_state.get("sales_staffing_events", []))
        tech_in_horizon, _ = _partition_events_by_horizon(st.session_state.get("tech_staffing_events_master", []), month_labels)
        sales_in_horizon, _ = _partition_events_by_horizon(st.session_state.get("sales_staffing_events_master", []), month_labels)
        st.session_state["tech_staffing_events"] = tech_in_horizon
        st.session_state["sales_staffing_events"] = sales_in_horizon

    if _show_section("Service and Replacement"):
        with st.expander("Service and Replacement", expanded=_section_expanded("Service and Replacement")):
            svc_left, svc_right = st.columns(2)
            with svc_left:
                st.number_input(
                    "Avg Service Ticket",
                    min_value=0.0,
                    key="avg_service_ticket",
                    help=help_with_guidance("avg_service_ticket", "Average revenue per service visit."),
                )
                st.slider(
                    "Service Material %",
                    0.0,
                    1.0,
                    step=0.001,
                    key="service_material_pct",
                    help=help_with_guidance("service_material_pct", "Material cost share applied to service revenue."),
                )
                st.slider("Attach Rate", 0.0, 1.0, step=0.001, key="attach_rate")
                st.number_input(
                    "Replacement Leads per Tech per Month",
                    min_value=0.0,
                    key="repl_leads_per_tech_per_month",
                    help=help_with_guidance("repl_leads_per_tech_per_month", "Replacement opportunities generated by each tech monthly."),
                )
                st.slider(
                    "Replacement Close Rate",
                    0.0,
                    1.0,
                    step=0.001,
                    key="repl_close_rate",
                    help=help_with_guidance("repl_close_rate", "Base replacement close rate before sales staffing lift."),
                )
                st.number_input("Sales Lift per FTE (Close Rate Points)", min_value=0.0, max_value=1.0, key="sales_repl_close_lift_per_fte")
                st.slider("Replacement Close Rate Cap", 0.0, 1.0, step=0.001, key="sales_repl_close_rate_cap")
            with svc_right:
                st.number_input(
                    "Avg Replacement Ticket",
                    min_value=0.0,
                    key="avg_repl_ticket",
                    help=help_with_guidance("avg_repl_ticket", "Average replacement job revenue."),
                )
                st.slider(
                    "Replacement Equipment %",
                    0.0,
                    1.0,
                    step=0.001,
                    key="repl_equipment_pct",
                    help=help_with_guidance("repl_equipment_pct", "Replacement equipment/material share of replacement revenue."),
                )
                st.number_input("Permit Cost per Replacement Job", min_value=0.0, key="permit_cost_per_repl_job")
                st.number_input("Disposal Cost per Replacement Job", min_value=0.0, key="disposal_cost_per_repl_job")
                st.slider("Financing Penetration", 0.0, 1.0, step=0.001, key="financing_penetration")
                st.slider("Financing Fee %", 0.0, 1.0, step=0.001, key="financing_fee_pct")
    if _show_section("Maintenance"):
        with st.expander("Maintenance", expanded=_section_expanded("Maintenance")):
            enable_maintenance = st.checkbox("Enable Maintenance", key="enable_maintenance")
            st.number_input(
                "Maintenance Capacity Visits per Tech per Month",
                min_value=0.0,
                key="maint_visits_capacity_per_tech_per_month",
                disabled=not enable_maintenance,
            )
            maint_res_col, maint_lc_col = st.columns(2)
            with maint_res_col:
                st.markdown("Residential Maintenance")
                st.number_input("Res Agreements Start", min_value=0.0, key="res_agreements_start", disabled=not enable_maintenance)
                st.number_input("Res Base New Agreements per Month", min_value=0.0, key="res_new_agreements_per_month", disabled=not enable_maintenance)
                st.slider("Res Churn Annual %", 0.0, 1.0, step=0.001, key="res_churn_annual_pct", disabled=not enable_maintenance)
                st.number_input("Res Monthly Agreement Fee", min_value=0.0, key="res_maint_monthly_fee", disabled=not enable_maintenance)
                st.number_input("Res Cost per Maintenance Visit", min_value=0.0, key="res_cost_per_maint_visit", disabled=not enable_maintenance)
                st.number_input("Res Visits per Agreement per Year", min_value=0.0, key="res_maint_visits_per_agreement_per_year", disabled=not enable_maintenance)
                st.slider("Res New Agreement Conversion per Call", 0.0, 1.0, step=0.001, key="res_maint_call_conversion_pct", disabled=not enable_maintenance)
                st.number_input("Res Agreements per Tech per Month Driver", min_value=0.0, key="res_maint_agreements_per_tech_per_month", disabled=not enable_maintenance)
                st.slider("Res Hybrid Weight: Call Driver", 0.0, 1.0, step=0.01, key="res_maint_hybrid_weight_calls", disabled=not enable_maintenance)
            with maint_lc_col:
                st.markdown("Light Commercial Maintenance")
                st.number_input("LC Agreements Start", min_value=0.0, key="lc_agreements_start", disabled=not enable_maintenance)
                st.number_input("LC Base New Agreements per Month", min_value=0.0, key="lc_new_agreements_per_month", disabled=not enable_maintenance)
                st.slider("LC Churn Annual %", 0.0, 1.0, step=0.001, key="lc_churn_annual_pct", disabled=not enable_maintenance)
                st.number_input("LC Quarterly Agreement Fee", min_value=0.0, key="lc_maint_quarterly_fee", disabled=not enable_maintenance)
                st.number_input("LC Cost per Maintenance Visit", min_value=0.0, key="lc_cost_per_maint_visit", disabled=not enable_maintenance)
                st.number_input("LC Visits per Agreement per Year", min_value=0.0, key="lc_maint_visits_per_agreement_per_year", disabled=not enable_maintenance)
                st.slider("LC New Agreement Conversion per Call", 0.0, 1.0, step=0.001, key="lc_maint_call_conversion_pct", disabled=not enable_maintenance)
                st.number_input("LC Agreements per Tech per Month Driver", min_value=0.0, key="lc_maint_agreements_per_tech_per_month", disabled=not enable_maintenance)
                st.slider("LC Hybrid Weight: Call Driver", 0.0, 1.0, step=0.01, key="lc_maint_hybrid_weight_calls", disabled=not enable_maintenance)
            if not enable_maintenance:
                st.caption("Maintenance assumptions are inactive while maintenance is disabled.")

    if _show_section("Upsell"):
        with st.expander("Upsell", expanded=_section_expanded("Upsell")):
            maintenance_enabled_for_upsell = bool(st.session_state.get("enable_maintenance", True))
            up_res_col, up_lc_col = st.columns(2)
            with up_res_col:
                st.markdown("Residential Service Upsell")
                st.slider("Res Service Upsell Conversion %", 0.0, 1.0, step=0.001, key="res_service_upsell_conversion_pct")
                st.number_input("Res Service Upsell Revenue per Converted Visit", min_value=0.0, key="res_service_upsell_revenue_per_visit")
                st.slider("Res Service Upsell Gross Margin %", 0.0, 1.0, step=0.001, key="res_service_upsell_gross_margin_pct")
                st.markdown("Residential Maintenance Upsell")
                st.slider(
                    "Res Maintenance Upsell Conversion %",
                    0.0,
                    1.0,
                    step=0.001,
                    key="res_maint_upsell_conversion_pct",
                    disabled=not maintenance_enabled_for_upsell,
                )
                st.number_input(
                    "Res Maintenance Upsell Revenue per Converted Visit",
                    min_value=0.0,
                    key="res_maint_upsell_revenue_per_visit",
                    disabled=not maintenance_enabled_for_upsell,
                )
                st.slider(
                    "Res Maintenance Upsell Gross Margin %",
                    0.0,
                    1.0,
                    step=0.001,
                    key="res_maint_upsell_gross_margin_pct",
                    disabled=not maintenance_enabled_for_upsell,
                )
            with up_lc_col:
                st.markdown("Light Commercial Service Upsell")
                st.slider("LC Service Upsell Conversion %", 0.0, 1.0, step=0.001, key="lc_service_upsell_conversion_pct")
                st.number_input("LC Service Upsell Revenue per Converted Visit", min_value=0.0, key="lc_service_upsell_revenue_per_visit")
                st.slider("LC Service Upsell Gross Margin %", 0.0, 1.0, step=0.001, key="lc_service_upsell_gross_margin_pct")
                st.markdown("Light Commercial Maintenance Upsell")
                st.slider(
                    "LC Maintenance Upsell Conversion %",
                    0.0,
                    1.0,
                    step=0.001,
                    key="lc_maint_upsell_conversion_pct",
                    disabled=not maintenance_enabled_for_upsell,
                )
                st.number_input(
                    "LC Maintenance Upsell Revenue per Converted Visit",
                    min_value=0.0,
                    key="lc_maint_upsell_revenue_per_visit",
                    disabled=not maintenance_enabled_for_upsell,
                )
                st.slider(
                    "LC Maintenance Upsell Gross Margin %",
                    0.0,
                    1.0,
                    step=0.001,
                    key="lc_maint_upsell_gross_margin_pct",
                    disabled=not maintenance_enabled_for_upsell,
                )
            if not maintenance_enabled_for_upsell:
                st.caption("Maintenance upsell assumptions are inactive while maintenance is disabled.")

    if st.session_state.get("res_new_build_install_schedule_master") is None:
        st.session_state["res_new_build_install_schedule_master"] = _sort_schedule(
            st.session_state.get("res_new_build_install_schedule", [])
        )
    if st.session_state.get("lc_new_build_install_schedule_master") is None:
        st.session_state["lc_new_build_install_schedule_master"] = _sort_schedule(
            st.session_state.get("lc_new_build_install_schedule", [])
        )
    res_sched_in_horizon, res_sched_out_of_horizon = _partition_schedule_by_horizon(
        st.session_state.get("res_new_build_install_schedule_master", []),
        month_labels,
    )
    lc_sched_in_horizon, lc_sched_out_of_horizon = _partition_schedule_by_horizon(
        st.session_state.get("lc_new_build_install_schedule_master", []),
        month_labels,
    )

    if _show_section("New-Build Installs"):
        with st.expander("New-Build Installs", expanded=_section_expanded("New-Build Installs")):
            new_build_mode = st.selectbox("New-Build Mode", ["base_seasonal", "schedule", "annual_total"], key="new_build_mode")
            st.slider("New-Build Seasonality Amplitude", 0.0, 0.4, step=0.005, key="new_build_seasonality_amplitude")
            nb_left, nb_right = st.columns(2)
            with nb_left:
                st.number_input(
                    "Res New-Build Installs per Month (Base)",
                    min_value=0.0,
                    key="res_new_build_installs_per_month",
                    disabled=new_build_mode != "base_seasonal",
                )
                st.number_input(
                    "Res New-Build Annual Installs",
                    min_value=0.0,
                    key="res_new_build_annual_installs",
                    disabled=new_build_mode != "annual_total",
                )
                st.number_input("Res New-Build Avg Ticket", min_value=0.0, key="res_new_build_avg_ticket")
                st.slider("Res New-Build Gross Margin %", 0.0, 1.0, step=0.001, key="res_new_build_gross_margin_pct")
            with nb_right:
                st.number_input(
                    "LC New-Build Installs per Month (Base)",
                    min_value=0.0,
                    key="lc_new_build_installs_per_month",
                    disabled=new_build_mode != "base_seasonal",
                )
                st.number_input(
                    "LC New-Build Annual Installs",
                    min_value=0.0,
                    key="lc_new_build_annual_installs",
                    disabled=new_build_mode != "annual_total",
                )
                st.number_input("LC New-Build Avg Ticket", min_value=0.0, key="lc_new_build_avg_ticket")
                st.slider("LC New-Build Gross Margin %", 0.0, 1.0, step=0.001, key="lc_new_build_gross_margin_pct")
            if new_build_mode in {"base_seasonal", "annual_total"}:
                st.caption(
                    "Schedule editors are enabled only in `schedule` mode. "
                    "Base and annual inputs are mode-specific and disabled when inactive."
                )
            if new_build_mode == "schedule":
                schedule_cfg = {
                    "month": st.column_config.SelectboxColumn("Month", options=month_labels, required=True),
                    "installs": st.column_config.NumberColumn("Installs", min_value=0.0, step=0.1),
                }
                nb_sched_left, nb_sched_right = st.columns(2)
                with nb_sched_left:
                    st.caption("Residential New-Build Install Schedule")
                    res_sched_df = st.data_editor(
                        _schedule_df(res_sched_in_horizon),
                        num_rows="dynamic",
                        column_config=schedule_cfg,
                        key="res_new_build_install_schedule_editor",
                        width="stretch",
                    )
                with nb_sched_right:
                    st.caption("Light Commercial New-Build Install Schedule")
                    lc_sched_df = st.data_editor(
                        _schedule_df(lc_sched_in_horizon),
                        num_rows="dynamic",
                        column_config=schedule_cfg,
                        key="lc_new_build_install_schedule_editor",
                        width="stretch",
                    )
                edited_res_sched_in_horizon = _sort_schedule(_schedule_from_editor(res_sched_df))
                edited_lc_sched_in_horizon = _sort_schedule(_schedule_from_editor(lc_sched_df))
                st.session_state["res_new_build_install_schedule_master"] = _merge_schedule_sets(
                    edited_res_sched_in_horizon,
                    res_sched_out_of_horizon,
                )
                st.session_state["lc_new_build_install_schedule_master"] = _merge_schedule_sets(
                    edited_lc_sched_in_horizon,
                    lc_sched_out_of_horizon,
                )
                st.session_state["res_new_build_install_schedule"] = edited_res_sched_in_horizon
                st.session_state["lc_new_build_install_schedule"] = edited_lc_sched_in_horizon
                if res_sched_out_of_horizon:
                    st.info(
                        f"Preserved {len(res_sched_out_of_horizon)} residential new-build schedule row(s) outside the current model horizon. "
                        "Increase horizon to view or edit them."
                    )
                if lc_sched_out_of_horizon:
                    st.info(
                        f"Preserved {len(lc_sched_out_of_horizon)} light-commercial new-build schedule row(s) outside the current model horizon. "
                        "Increase horizon to view or edit them."
                    )
            else:
                st.session_state["res_new_build_install_schedule"] = res_sched_in_horizon
                st.session_state["lc_new_build_install_schedule"] = lc_sched_in_horizon
    else:
        st.session_state["res_new_build_install_schedule"] = res_sched_in_horizon
        st.session_state["lc_new_build_install_schedule"] = lc_sched_in_horizon

    if _show_section("Marketing and Fleet"):
        with st.expander("Marketing and Fleet", expanded=_section_expanded("Marketing and Fleet")):
            marketing_col, fleet_col = st.columns(2)
            with marketing_col:
                st.markdown("Marketing")
                paid_leads_mode = st.selectbox("Paid Leads Mode", ["fixed", "per_tech"], key="paid_leads_mode")
                st.number_input(
                    "Paid Leads per Month",
                    min_value=0.0,
                    key="paid_leads_per_month",
                    disabled=paid_leads_mode != "fixed",
                )
                st.number_input(
                    "Paid Leads per Tech per Month",
                    min_value=0.0,
                    key="paid_leads_per_tech_per_month",
                    disabled=paid_leads_mode != "per_tech",
                )
                st.number_input(
                    "Cost per Lead",
                    min_value=0.0,
                    key="cost_per_lead",
                    help=help_with_guidance("cost_per_lead", "Average paid lead acquisition cost."),
                )
                st.number_input(
                    "Branding Fixed Monthly",
                    min_value=0.0,
                    key="branding_fixed_monthly",
                    help=help_with_guidance("branding_fixed_monthly", "Fixed monthly branding or awareness spend."),
                )
            with fleet_col:
                st.markdown("Fleet")
                st.number_input("Trucks per Tech", min_value=0.0, key="trucks_per_tech", help=help_with_guidance("trucks_per_tech", "Truck deployment ratio by technician count."))
                st.number_input(
                    "Truck Payment Monthly",
                    min_value=0.0,
                    key="truck_payment_monthly",
                    help=help_with_guidance("truck_payment_monthly", "Monthly truck financing payment per active truck."),
                )
                st.number_input("Fuel per Truck Monthly", min_value=0.0, key="fuel_per_truck_monthly")
                st.number_input("Maintenance per Truck Monthly", min_value=0.0, key="maint_per_truck_monthly")
                st.number_input("Truck Insurance per Truck Monthly", min_value=0.0, key="truck_insurance_per_truck_monthly")
                st.number_input(
                    "Truck Purchase Price",
                    min_value=0.0,
                    key="truck_purchase_price",
                    help=help_with_guidance("truck_purchase_price", "Purchase price used for truck downpayment capex."),
                )
                capex_trucks_mode = st.selectbox("Capex Trucks Mode", ["payments_only", "purchase_with_downpayment"], key="capex_trucks_mode")
                st.slider(
                    "Truck Downpayment %",
                    0.0,
                    1.0,
                    step=0.001,
                    key="truck_downpayment_pct",
                    disabled=capex_trucks_mode != "purchase_with_downpayment",
                )
                st.slider("Truck Financed %", 0.0, 1.0, step=0.001, key="truck_financed_pct")

    if _show_section("Overhead and Management"):
        with st.expander("Overhead and Management", expanded=_section_expanded("Overhead and Management")):
            oh_col, mgmt_col = st.columns(2)
            with oh_col:
                st.markdown("Overhead")
                st.number_input(
                    "Office Payroll Monthly",
                    min_value=0.0,
                    key="office_payroll_monthly",
                    help=help_with_guidance("office_payroll_monthly", "Non-field payroll in overhead."),
                )
                st.number_input("Rent Monthly", min_value=0.0, key="rent_monthly")
                st.number_input("Utilities Monthly", min_value=0.0, key="utilities_monthly")
                st.number_input("Insurance Monthly", min_value=0.0, key="insurance_monthly")
                st.number_input("Software Monthly", min_value=0.0, key="software_monthly")
                st.number_input("Other Fixed Monthly", min_value=0.0, key="other_fixed_monthly")
            with mgmt_col:
                st.markdown("Management Payroll")
                st.number_input(
                    "Manager Salary Monthly",
                    min_value=0.0,
                    key="manager_salary_monthly",
                    help=help_with_guidance("manager_salary_monthly", "Monthly salary recognized once manager start date is reached."),
                )
                st.number_input(
                    "Ops Manager Salary Monthly",
                    min_value=0.0,
                    key="ops_manager_salary_monthly",
                    help=help_with_guidance("ops_manager_salary_monthly", "Monthly salary recognized once ops manager start date is reached."),
                )
                st.number_input(
                    "Marketing Manager Salary Monthly",
                    min_value=0.0,
                    key="marketing_manager_salary_monthly",
                    help=help_with_guidance("marketing_manager_salary_monthly", "Monthly salary recognized once marketing manager start date is reached."),
                )
                st.number_input("Manager Start Year", min_value=2000, max_value=2100, step=1, key="manager_start_year")
                m1 = st.selectbox("Manager Start Month", month_opts, index=int(st.session_state["manager_start_month"]) - 1, key="manager_start_month_opt")
                st.session_state["manager_start_month"] = parse_month_name_option(m1)
                st.number_input("Ops Manager Start Year", min_value=2000, max_value=2100, step=1, key="ops_manager_start_year")
                m2 = st.selectbox("Ops Manager Start Month", month_opts, index=int(st.session_state["ops_manager_start_month"]) - 1, key="ops_manager_start_month_opt")
                st.session_state["ops_manager_start_month"] = parse_month_name_option(m2)
                st.number_input("Marketing Manager Start Year", min_value=2000, max_value=2100, step=1, key="marketing_manager_start_year")
                m3 = st.selectbox(
                    "Marketing Manager Start Month",
                    month_opts,
                    index=int(st.session_state["marketing_manager_start_month"]) - 1,
                    key="marketing_manager_start_month_opt",
                )
                st.session_state["marketing_manager_start_month"] = parse_month_name_option(m3)

            st.markdown("Raises and Escalation")
            raise_month_opt = st.selectbox("Raise Effective Month", month_opts, index=int(st.session_state["raise_effective_month"]) - 1, key="raise_effective_month_opt")
            st.session_state["raise_effective_month"] = parse_month_name_option(raise_month_opt)
            st.slider(
                "Annual Raise % - Tech",
                0.0,
                0.2,
                step=0.001,
                key="annual_raise_pct_tech",
                help=help_with_guidance("annual_raise_pct_tech", "Annual wage step-up for technician labor."),
            )
            st.slider(
                "Annual Raise % - Sales",
                0.0,
                0.2,
                step=0.001,
                key="annual_raise_pct_sales",
                help=help_with_guidance("annual_raise_pct_sales", "Annual wage step-up for sales labor."),
            )
            st.slider("Annual Raise % - Manager", 0.0, 0.2, step=0.001, key="annual_raise_pct_manager")
            st.slider("Annual Raise % - Ops Manager", 0.0, 0.2, step=0.001, key="annual_raise_pct_ops_manager")
            st.slider("Annual Raise % - Marketing Manager", 0.0, 0.2, step=0.001, key="annual_raise_pct_marketing_manager")

    if _show_section("Working Capital, Debt, and Distributions"):
        with st.expander("Working Capital, Debt, and Distributions", expanded=_section_expanded("Working Capital, Debt, and Distributions")):
            wc_col, debt_col = st.columns(2)
            with wc_col:
                st.markdown("Working Capital")
                st.number_input("AR Days", min_value=0.0, key="ar_days")
                st.number_input("AP Days", min_value=0.0, key="ap_days")
                st.number_input("Inventory Days", min_value=0.0, key="inventory_days")
                st.number_input("Opening AR Balance", min_value=0.0, key="opening_ar_balance")
                st.number_input("Opening AP Balance", min_value=0.0, key="opening_ap_balance")
                st.number_input("Opening Inventory Balance", min_value=0.0, key="opening_inventory_balance")
                st.number_input("Starting Cash", min_value=0.0, key="starting_cash")
            with debt_col:
                st.markdown("Debt")
                enable_term_loan = st.checkbox("Enable Term Loan", key="enable_term_loan")
                st.number_input("Loan Principal", min_value=0.0, key="loan_principal", disabled=not enable_term_loan)
                st.slider(
                    "Loan Annual Rate",
                    0.0,
                    1.0,
                    step=0.001,
                    key="loan_annual_rate",
                    help=help_with_guidance("loan_annual_rate", "Annualized term debt rate."),
                    disabled=not enable_term_loan,
                )
                st.number_input("Loan Term Months", min_value=1, step=1, key="loan_term_months", disabled=not enable_term_loan)
                enable_loc = st.checkbox("Enable LOC", key="enable_loc")
                st.number_input("LOC Limit", min_value=0.0, key="loc_limit", disabled=not enable_loc)
                st.slider(
                    "LOC Annual Rate",
                    0.0,
                    1.0,
                    step=0.001,
                    key="loc_annual_rate",
                    help=help_with_guidance("loc_annual_rate", "Annualized line-of-credit borrowing rate."),
                    disabled=not enable_loc,
                )
                st.number_input(
                    "Min Cash Target",
                    min_value=0.0,
                    key="min_cash_target",
                    help=help_with_guidance("min_cash_target", "Cash floor to protect via operating and LOC policy."),
                    disabled=not enable_loc,
                )
                st.number_input("LOC Repay Buffer", min_value=0.0, key="loc_repay_buffer", disabled=not enable_loc)
            st.markdown("Distributions")
            enable_distributions = st.checkbox("Enable Distributions", key="enable_distributions")
            st.slider(
                "Distributions % of EBITDA",
                0.0,
                1.0,
                step=0.001,
                key="distributions_pct_of_ebitda",
                disabled=not enable_distributions,
            )
            st.number_input(
                "Distributions Only if Cash Above",
                min_value=0.0,
                key="distributions_only_if_cash_above",
                disabled=not enable_distributions,
            )
            if not enable_term_loan and not enable_loc:
                st.caption("Debt assumptions are inactive while term loan and LOC are both disabled.")
current_assumptions_state = _assumptions_from_state()
current_assumptions_json = _serialize_assumptions(current_assumptions_state)
if not st.session_state.get("applied_assumptions_json"):
    st.session_state["applied_assumptions_json"] = current_assumptions_json

try:
    applied_assumptions_state = json.loads(st.session_state["applied_assumptions_json"])
except (TypeError, ValueError, json.JSONDecodeError):
    st.session_state["applied_assumptions_json"] = current_assumptions_json
    applied_assumptions_state = current_assumptions_state

pending_change_keys = _changed_assumption_keys(current_assumptions_state, applied_assumptions_state)
if (st.session_state.get("auto_run_model", True) and pending_change_keys) or (manual_apply_inputs and pending_change_keys):
    st.session_state["applied_assumptions_json"] = current_assumptions_json
    applied_assumptions_state = current_assumptions_state
    pending_change_keys = []

assumptions, schema_warnings, unknown_keys = migrate_assumptions(applied_assumptions_state)
assumptions, normalization_warnings = _normalize_inputs_for_runtime(assumptions)
input_warnings: list[str] = []
input_warnings.extend(schema_warnings)
if unknown_keys:
    input_warnings.append(f"Ignored unknown keys: {', '.join(unknown_keys)}")
input_warnings.extend(advisory_warnings(assumptions))
input_warnings.extend(normalization_warnings)
input_warnings = list(dict.fromkeys([w for w in input_warnings if str(w).strip()]))

try:
    assumptions_json = _serialize_assumptions(assumptions)
    nominal_df = _run_model_cached(assumptions_json)
except ValueError as exc:
    append_runtime_event(
        level="WARNING",
        event="model_validation_recovered",
        message="Model validation failed on initial run and was auto-repaired.",
        context={"error": str(exc)},
        exc=exc,
    )
    repaired, repair_warnings = _normalize_inputs_for_runtime(assumptions)
    input_warnings.extend(repair_warnings)
    input_warnings.append(f"Validation warning recovered automatically: {exc}")
    try:
        assumptions = repaired
        assumptions_json = _serialize_assumptions(assumptions)
        nominal_df = _run_model_cached(assumptions_json)
    except ValueError as exc2:
        append_runtime_event(
            level="ERROR",
            event="model_validation_failed",
            message="Model failed validation after auto-repair.",
            context={"error": str(exc2), "repair_warnings": repair_warnings},
            exc=exc2,
        )
        st.error(f"Input validation error: {exc2}")
        st.stop()

input_warning_signature = _stable_json(input_warnings)
if input_warnings and st.session_state.get("_input_warning_log_signature") != input_warning_signature:
    append_runtime_event(
        level="WARNING",
        event="input_warnings",
        message=f"{len(input_warnings)} input warning(s) generated during model run.",
        context={"warnings": input_warnings},
    )
    st.session_state["_input_warning_log_signature"] = input_warning_signature
elif not input_warnings:
    st.session_state["_input_warning_log_signature"] = ""

if input_warnings:
    with st.expander(f"[!] Input Warnings ({len(input_warnings)})", expanded=False):
        st.caption("Model calculations continue using sanitized values where necessary.")
        for warning in input_warnings:
            st.write(f"- {warning}")

integrity_findings = run_integrity_checks(nominal_df, assumptions, tol=1e-3)
integrity_signature = _stable_json(integrity_findings)
if integrity_findings and st.session_state.get("_integrity_log_signature") != integrity_signature:
    append_runtime_event(
        level="ERROR",
        event="integrity_checks_failed",
        message=f"{len(integrity_findings)} integrity check(s) failed.",
        context={"finding_count": len(integrity_findings), "findings": integrity_findings[:25]},
    )
    st.session_state["_integrity_log_signature"] = integrity_signature
elif not integrity_findings:
    st.session_state["_integrity_log_signature"] = ""

if integrity_findings:
    integrity_df = pd.DataFrame(integrity_findings)
    with st.expander(f"[!] Accounting Integrity Findings ({len(integrity_df)})", expanded=False):
        st.caption("Reconcile these deltas before relying on outputs for release decisions.")
        st.dataframe(_format_dataframe_for_display(integrity_df), width="stretch", hide_index=True)
        st.download_button(
            "Download Integrity Findings CSV",
            integrity_df.to_csv(index=False),
            file_name="hvac_integrity_findings.csv",
            mime="text/csv",
        )
else:
    st.caption("Accounting integrity checks: passed.")

if not st.session_state.get("auto_run_model", True) and pending_change_keys:
    st.info(
        f"Manual apply mode: {len(pending_change_keys)} input change(s) are pending. "
        "Outputs below reflect the last applied state."
    )

for dframe in (nominal_df,):
    dframe.attrs["attach_rate"] = assumptions["attach_rate"]
    dframe.attrs["ar_days"] = assumptions["ar_days"]
    dframe.attrs["ap_days"] = assumptions["ap_days"]
    dframe.attrs["inventory_days"] = assumptions["inventory_days"]

view_df = apply_value_mode(nominal_df, assumptions, assumptions["value_mode"])
for dframe in (view_df,):
    dframe.attrs["attach_rate"] = assumptions["attach_rate"]
    dframe.attrs["ar_days"] = assumptions["ar_days"]
    dframe.attrs["ap_days"] = assumptions["ap_days"]
    dframe.attrs["inventory_days"] = assumptions["inventory_days"]

labels = view_df["Year_Month_Label"].tolist()
if not st.session_state["range_start_label"] or st.session_state["range_start_label"] not in labels:
    st.session_state["range_start_label"] = labels[0]
if not st.session_state["range_end_label"] or st.session_state["range_end_label"] not in labels:
    st.session_state["range_end_label"] = labels[-1]

with st.sidebar:
    st.subheader("Time Range")
    st.selectbox(
        "Range Preset",
        ["Full horizon", "Year 1", "Last 12 months", "Rolling 24 months", "Custom"],
        key="range_preset",
    )
    range_start_default = st.session_state["range_start_label"]
    range_end_default = st.session_state["range_end_label"]
    if st.session_state["range_preset"] == "Full horizon":
        range_start_default = labels[0]
        range_end_default = labels[-1]
    elif st.session_state["range_preset"] == "Year 1":
        yr1 = view_df[view_df["Year"] == 1]["Year_Month_Label"].tolist()
        range_start_default = yr1[0]
        range_end_default = yr1[-1]
    elif st.session_state["range_preset"] == "Last 12 months":
        range_start_default = labels[max(0, len(labels) - 12)]
        range_end_default = labels[-1]
    elif st.session_state["range_preset"] == "Rolling 24 months":
        range_start_default = labels[max(0, len(labels) - 24)]
        range_end_default = labels[-1]

    if labels.index(range_start_default) > labels.index(range_end_default):
        range_start_default, range_end_default = range_end_default, range_start_default

    selected_range = st.select_slider(
        "Range Window",
        options=labels,
        value=(range_start_default, range_end_default),
        disabled=st.session_state["range_preset"] != "Custom",
        help="Select the analysis window used across KPIs, charts, tables, and goal seek.",
    )
    st.session_state["range_start_label"], st.session_state["range_end_label"] = selected_range
    range_start_idx = labels.index(st.session_state["range_start_label"])
    range_end_idx = labels.index(st.session_state["range_end_label"])
    st.caption(f"Selected months: {abs(range_end_idx - range_start_idx) + 1}")

range_df = _filter_df(view_df, st.session_state["range_start_label"], st.session_state["range_end_label"])
metrics_full = compute_metrics(view_df, assumptions["horizon_months"])
metrics_range = compute_metrics(range_df, len(range_df))
annual_kpis_full = _build_annual_kpis(view_df, metrics_full)
annual_kpis_range = _build_annual_kpis(range_df, metrics_range)

with st.sidebar:
    st.subheader("Goal Seek")
    goal_metric_options = [
        "Total Revenue",
        "Total EBITDA",
        "Total Free Cash Flow",
        "Minimum Ending Cash",
        "Negative Cash Months",
        "Avg Gross Margin %",
        "Year Total Revenue",
        "Year EBITDA",
        "Year Free Cash Flow",
        "Column Aggregate",
    ]
    st.selectbox("Target Metric", goal_metric_options, key="goal_target_metric")
    selected_goal_metric = st.session_state["goal_target_metric"]
    available_years = sorted(range_df["Year"].unique().tolist())
    year_metric_options = {"Year Total Revenue", "Year EBITDA", "Year Free Cash Flow"}
    if available_years and selected_goal_metric in year_metric_options:
        if st.session_state["goal_year"] not in available_years:
            st.session_state["goal_year"] = available_years[0]
        st.selectbox("Target Year", available_years, key="goal_year")
    elif selected_goal_metric in year_metric_options:
        st.caption("No valid years in selected range.")
    numeric_cols = [c for c in range_df.columns if pd.api.types.is_numeric_dtype(range_df[c])]
    if selected_goal_metric == "Column Aggregate":
        st.selectbox("Column (for Column Aggregate)", numeric_cols, key="goal_column_name")
        st.selectbox("Aggregation", ["sum", "avg", "min", "max", "end"], key="goal_column_agg")
    st.number_input("Target Value", key="goal_target_value")

    adjustable_inputs = available_sensitivity_drivers(assumptions)
    default_adjust = st.session_state.get("goal_adjustable_input", "avg_service_ticket")
    if default_adjust not in adjustable_inputs and adjustable_inputs:
        default_adjust = adjustable_inputs[0]
        st.session_state["goal_adjustable_input"] = default_adjust
    if adjustable_inputs:
        st.selectbox("Adjustable Input", adjustable_inputs, key="goal_adjustable_input")
    else:
        st.warning("No eligible scalar numeric inputs are available for goal seek.")
        st.session_state["goal_adjustable_input"] = ""

    g = INPUT_GUIDANCE.get(st.session_state["goal_adjustable_input"], {})
    current_val = float(assumptions.get(st.session_state["goal_adjustable_input"], 0.0)) if st.session_state["goal_adjustable_input"] else 0.0
    bound_low_default = float(g.get("min", 0.0))
    bound_high_default = float(g.get("max", max(1.0, current_val * 3.0 + 1)))
    bound_low = st.number_input("Solver Lower Bound", value=bound_low_default, min_value=0.0, step=0.01)
    bound_high = st.number_input("Solver Upper Bound", value=max(bound_high_default, bound_low + 0.01), min_value=bound_low + 0.01, step=0.01)
    run_goal_seek = st.button(
        "Run Goal Seek",
        disabled=not adjustable_inputs,
        help="Solve for the selected input value needed to hit the target metric within the specified bounds.",
    )

    if run_goal_seek:
        metric_name = st.session_state["goal_target_metric"]
        column_name = st.session_state["goal_column_name"]
        agg = st.session_state["goal_column_agg"]
        goal_year = int(st.session_state["goal_year"])
        target_value = float(st.session_state["goal_target_value"])
        input_key = st.session_state["goal_adjustable_input"]

        def evaluator(x: float) -> float:
            scenario = deepcopy(assumptions)
            scenario[input_key] = float(x)
            scenario, _, _ = migrate_assumptions(scenario)
            ndf = _run_model_cached(_serialize_assumptions(scenario))
            vdf = apply_value_mode(ndf, scenario, scenario["value_mode"])
            fdf = _filter_df(vdf, st.session_state["range_start_label"], st.session_state["range_end_label"])
            return _metric_from_selection(fdf, metric_name, column_name, agg, goal_year)

        result = solve_bounded_scalar(evaluator, target_value, bound_low, bound_high, tol=1e-2, max_iter=80)
        st.session_state["goal_seek_result"] = {
            "status": result.status,
            "value": result.value,
            "achieved": result.achieved,
            "message": result.message,
            "iterations": result.iterations,
        }
        if result.status != "solved":
            append_runtime_event(
                level="WARNING",
                event="goal_seek_failed",
                message=result.message,
                context={
                    "metric": metric_name,
                    "input_key": input_key,
                    "target_value": target_value,
                    "lower_bound": bound_low,
                    "upper_bound": bound_high,
                    "status": result.status,
                    "iterations": result.iterations,
                },
            )

    gs = st.session_state.get("goal_seek_result")
    if gs:
        if gs["status"] == "solved":
            st.success(
                f"Goal seek solved in {gs['iterations']} iterations: {st.session_state['goal_adjustable_input']}={gs['value']:.4f}, achieved={gs['achieved']:.4f}"
            )
        else:
            st.warning(gs["message"])

    st.subheader("AI Export Pack")
    st.caption(
        "Create a ChatGPT-ready JSON package with assumptions, outputs, transformation logic, and optional source-code context."
    )
    st.radio(
        "Export Scope",
        AI_EXPORT_SCOPE_OPTIONS,
        key="ai_export_scope",
        help="Use active-only for current what-if analysis, or include multiple saved scenarios/workspaces for cross-scenario AI analysis.",
    )
    saved_scenario_names = list_saved_names(SCENARIO_TYPE)
    saved_workspace_names = list_saved_names(WORKSPACE_TYPE)
    selected_ai_scenarios: list[str] = []
    selected_ai_workspaces: list[str] = []
    if st.session_state["ai_export_scope"] in {
        "Saved scenarios (select multiple)",
        "Active + selected saved items",
    }:
        selected_ai_scenarios = st.multiselect(
            "Saved scenarios to include",
            options=saved_scenario_names,
            key="ai_export_selected_scenarios",
        )
    if st.session_state["ai_export_scope"] in {
        "Saved workspaces (select multiple)",
        "Active + selected saved items",
    }:
        selected_ai_workspaces = st.multiselect(
            "Saved workspaces to include",
            options=saved_workspace_names,
            key="ai_export_selected_workspaces",
        )

    st.toggle("Include source-code snapshot", key="ai_export_include_source_code")
    st.toggle("Include runtime log tail", key="ai_export_include_runtime_logs")
    st.number_input(
        "Runtime log rows for export",
        min_value=0,
        max_value=2000,
        step=20,
        key="ai_export_log_limit",
        disabled=not st.session_state["ai_export_include_runtime_logs"],
    )

    generate_ai_export = st.button(
        "Generate AI Export Pack",
        help="Build a comprehensive JSON package suitable for upload to AI assistants for deep analysis and troubleshooting.",
    )
    if generate_ai_export:
        scope = st.session_state["ai_export_scope"]
        needs_saved_scenarios = scope in {"Saved scenarios (select multiple)", "Active + selected saved items"}
        needs_saved_workspaces = scope in {"Saved workspaces (select multiple)", "Active + selected saved items"}
        if needs_saved_scenarios and not selected_ai_scenarios and not (
            scope == "Active + selected saved items" and selected_ai_workspaces
        ):
            st.warning("Select at least one saved scenario or choose a scope that includes only active data.")
        elif needs_saved_workspaces and not selected_ai_workspaces and not (
            scope == "Active + selected saved items" and selected_ai_scenarios
        ):
            st.warning("Select at least one saved workspace or choose a scope that includes only active data.")
        else:
            with st.spinner("Building AI export pack..."):
                try:
                    pack = _build_ai_export_pack(
                        scope=scope,
                        active_assumptions=assumptions,
                        active_ui_state=_current_ui_state(),
                        active_range=(st.session_state["range_start_label"], st.session_state["range_end_label"]),
                        selected_scenarios=selected_ai_scenarios,
                        selected_workspaces=selected_ai_workspaces,
                        include_source_code=bool(st.session_state["ai_export_include_source_code"]),
                        include_runtime_logs=bool(st.session_state["ai_export_include_runtime_logs"]),
                        runtime_log_limit=int(st.session_state["ai_export_log_limit"]),
                        active_input_warnings=input_warnings,
                        active_integrity_findings=integrity_findings,
                        active_goal_seek_result=st.session_state.get("goal_seek_result"),
                    )
                except Exception as exc:
                    append_runtime_event(
                        level="ERROR",
                        event="ai_export_build_failed",
                        message="Failed to build AI export pack.",
                        context={"scope": scope},
                        exc=exc,
                    )
                    st.error(f"AI export failed: {exc}")
                else:
                    payload_json = json.dumps(pack, indent=2, ensure_ascii=False, default=_json_default)
                    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
                    st.session_state["ai_export_payload_json"] = payload_json
                    st.session_state["ai_export_filename"] = f"hvac_ai_export_{timestamp}.json"
                    st.session_state["ai_export_summary"] = (
                        f"{pack['summary']['scenario_count']} context(s), {pack['summary']['failure_count']} failure(s)."
                    )
                    st.success(f"AI export ready. {st.session_state['ai_export_summary']}")

    ai_payload = st.session_state.get("ai_export_payload_json", "")
    if ai_payload:
        payload_kb = len(ai_payload.encode("utf-8")) / 1024.0
        st.caption(
            f"Prepared pack: `{st.session_state.get('ai_export_filename', 'hvac_ai_export.json')}` "
            f"({payload_kb:,.1f} KB)."
        )
        st.download_button(
            "Download AI Export JSON",
            ai_payload,
            file_name=st.session_state.get("ai_export_filename", "hvac_ai_export.json"),
            mime="application/json",
            help="Upload this JSON directly to ChatGPT or other AI tools for context-rich analysis.",
        )

if st.session_state["autosave_enabled"] and st.session_state.get("active_workspace_name"):
    autosave_name = st.session_state["active_workspace_name"]
    autosave_bundle = build_workspace_bundle(autosave_name, assumptions, _current_ui_state())
    autosave_ok, autosave_msg = save_named_bundle(WORKSPACE_TYPE, autosave_name, autosave_bundle, overwrite=True)
    if not autosave_ok:
        append_runtime_event(
            level="WARNING",
            event="autosave_workspace_failed",
            message=autosave_msg,
            context={"workspace_name": autosave_name},
        )

st.caption(f"Display mode: {value_mode_label(assumptions['value_mode'])}")
status_1, status_2, status_3, status_4 = st.columns(4)
status_1.metric("Run Mode", "Live" if st.session_state.get("auto_run_model", True) else "Manual Apply")
status_2.metric("Pending Changes", len(pending_change_keys))
status_3.metric("Selected Range Months", len(range_df))
status_4.metric(
    "Range",
    f"{st.session_state['range_start_label']} to {st.session_state['range_end_label']}",
)
_display_help_panel()

summary_tab, cashflow_tab, drivers_tab, sens_tab = st.tabs(
    ["Summary Dashboard", "Line-by-line Cash Flow", "Drivers and Custom Visuals", "Sensitivity"]
)

with summary_tab:
    st.subheader("Headline KPIs (Selected Range with Full-Horizon Reference)")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Revenue", f"${range_df['Total Revenue'].sum():,.0f}", f"Full ${view_df['Total Revenue'].sum():,.0f}")
    c2.metric("Total EBITDA", f"${range_df['EBITDA'].sum():,.0f}", f"Full ${view_df['EBITDA'].sum():,.0f}")
    c3.metric("Total Free Cash Flow", f"${range_df['Free Cash Flow'].sum():,.0f}", f"Full ${view_df['Free Cash Flow'].sum():,.0f}")
    c4.metric(
        "Minimum Ending Cash",
        f"${metrics_range['minimum_ending_cash']:,.0f}",
        f"Full ${metrics_full['minimum_ending_cash']:,.0f}",
    )
    c5.metric("Negative Cash Months", f"{metrics_range['negative_cash_months']}", f"Full {metrics_full['negative_cash_months']}")
    c6.metric("Avg Gross Margin", f"{100 * metrics_range['gross_margin_full_period_avg']:.1f}%")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Cash Conversion Cycle", f"{metrics_range['ccc']:.1f} days")
    d2.metric("CAC", f"${metrics_range['cac']:,.2f}")
    d3.metric("Break-even Revenue", f"${metrics_range['break_even_revenue']:,.0f}")
    d4.metric("Total Disbursements", f"${metrics_range['total_disbursements']:,.0f}")

    st.caption("Annual KPIs for selected range")
    st.dataframe(_format_dataframe_for_display(annual_kpis_range), width="stretch")
    with st.expander("Full-Horizon Annual KPI Reference", expanded=False):
        st.dataframe(_format_dataframe_for_display(annual_kpis_full), width="stretch")

    seg = range_df[
        ["Date", "Service Revenue", "Replacement Revenue", "Maintenance Revenue", "Upsell Revenue", "New Build Revenue"]
    ].melt("Date", var_name="Segment", value_name="Revenue")
    st.plotly_chart(px.area(seg, x="Date", y="Revenue", color="Segment", title="Revenue by Segment"), width="stretch")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=range_df["Date"], y=range_df["EBITDA"], name="EBITDA"))
    fig.add_trace(
        go.Scatter(
            x=range_df["Date"],
            y=100 * range_df["EBITDA"] / range_df["Total Revenue"].replace(0, pd.NA),
            name="EBITDA Margin %",
            yaxis="y2",
        )
    )
    fig.update_layout(title="EBITDA and EBITDA Margin", yaxis2=dict(overlaying="y", side="right"))
    st.plotly_chart(fig, width="stretch")

    st.plotly_chart(px.line(range_df, x="Date", y="End Cash", title="Ending Cash Balance"), width="stretch")

    gm = 100 * range_df["Gross Profit"] / range_df["Total Revenue"].replace(0, pd.NA)
    op = 100 * range_df["Total OPEX"] / range_df["Total Revenue"].replace(0, pd.NA)
    gm_df = pd.DataFrame({"Date": range_df["Date"], "Gross Margin %": gm, "OPEX % of Revenue": op})
    gm_melt = gm_df.melt("Date", var_name="Metric", value_name="Percent")
    st.plotly_chart(px.line(gm_melt, x="Date", y="Percent", color="Metric", title="Gross Margin % and OPEX % of Revenue"), width="stretch")
with cashflow_tab:
    st.subheader("Monthly line-by-line cash flow")
    selected_month = st.selectbox("Select Month for Cash Flow Bridge", options=range_df["Year_Month_Label"].tolist(), index=len(range_df) - 1)
    row = range_df.loc[range_df["Year_Month_Label"] == selected_month].iloc[0]

    grouped_subtotals = pd.DataFrame(
        [
            {"Section": "A Operating", "Line Item": "EBITDA", "Amount": row["EBITDA"]},
            {"Section": "A Operating", "Line Item": "- Change in NWC", "Amount": -row["Change in NWC"]},
            {"Section": "A Operating", "Line Item": "Operating Cash Flow", "Amount": row["Operating Cash Flow"]},
            {"Section": "B Investing", "Line Item": "- Gross Capex", "Amount": -row["Gross Capex"]},
            {"Section": "B Investing", "Line Item": "+ Asset Salvage Proceeds", "Amount": row["Asset Salvage Proceeds"]},
            {"Section": "B Investing", "Line Item": "Capex (Net)", "Amount": row["Capex"]},
            {"Section": "B Investing", "Line Item": "Free Cash Flow", "Amount": row["Free Cash Flow"]},
            {"Section": "C Financing", "Line Item": "- Term Loan Payment", "Amount": -row["Term Loan Payment"]},
            {"Section": "C Financing", "Line Item": "- LOC Interest", "Amount": -row["LOC Interest"]},
            {"Section": "C Financing", "Line Item": "+ LOC Draw", "Amount": row["LOC Draw"]},
            {"Section": "C Financing", "Line Item": "- LOC Repay", "Amount": -row["LOC Repay"]},
            {"Section": "C Financing", "Line Item": "- Owner Distributions", "Amount": -row["Owner Distributions"]},
            {"Section": "Total", "Line Item": "Net Cash Flow", "Amount": row["Net Cash Flow"]},
        ]
    )
    st.dataframe(_format_dataframe_for_display(grouped_subtotals), width="stretch", hide_index=True)

    waterfall = go.Figure(
        go.Waterfall(
            name="Cash Flow Bridge",
            orientation="v",
            measure=[
                "absolute",
                "relative",
                "total",
                "relative",
                "total",
                "relative",
                "relative",
                "total",
                "relative",
                "relative",
                "relative",
                "relative",
                "relative",
                "total",
            ],
            x=[
                "Total Revenue",
                "Total Direct Costs",
                "Gross Profit",
                "Total OPEX",
                "EBITDA",
                "Change in NWC",
                "Capex",
                "Free Cash Flow",
                "Term Loan Payment",
                "LOC Interest",
                "LOC Draw",
                "LOC Repay",
                "Owner Distributions",
                "Net Cash Flow",
            ],
            y=[
                row["Total Revenue"],
                -row["Total Direct Costs"],
                0,
                -row["Total OPEX"],
                0,
                -row["Change in NWC"],
                -row["Capex"],
                0,
                -row["Term Loan Payment"],
                -row["LOC Interest"],
                row["LOC Draw"],
                -row["LOC Repay"],
                -row["Owner Distributions"],
                0,
            ],
        )
    )
    waterfall.update_layout(title=f"Monthly Cash Flow Bridge ({selected_month})", showlegend=False)
    st.plotly_chart(waterfall, width="stretch")

    disbursement_columns = [
        "Date",
        "Capex",
        "Term Loan Payment",
        "LOC Interest",
        "LOC Repay",
        "Owner Distributions",
        "Total Disbursements",
    ]
    if st.button("Use Disbursement Column Set", help="Quick-select a default set of disbursement-related columns for the cashflow table."):
        st.session_state["table_selected_columns"] = disbursement_columns

    selectable_cols = view_df.columns.tolist()
    default_cols = st.session_state.get("table_selected_columns") or [
        "Date",
        "Total Revenue",
        "EBITDA",
        "Free Cash Flow",
        "End Cash",
        "Total Disbursements",
    ]
    st.session_state["table_selected_columns"] = st.multiselect(
        "Displayed table columns",
        selectable_cols,
        default=[c for c in default_cols if c in selectable_cols],
    )
    table_cols = st.session_state["table_selected_columns"] or selectable_cols
    st.dataframe(_format_dataframe_for_display(range_df[table_cols]), width="stretch")
    st.download_button(
        "Download CSV (Selected Range)",
        range_df[table_cols].to_csv(index=False),
        file_name="hvac_cashflow_range.csv",
        mime="text/csv",
        help="Download the currently filtered range and selected table columns as CSV.",
    )

with drivers_tab:
    st.subheader("Core drivers")
    drv = range_df[
        [
            "Date",
            "Techs",
            "Sales Staff",
            "Trucks",
            "Retained Trucks",
            "Calls",
            "Replacement Leads",
            "Res Maintenance Agreements",
            "LC Maintenance Agreements",
        ]
    ]
    fleet_drv = drv.copy()
    fleet_drv["Total Fleet Units"] = fleet_drv["Trucks"] + fleet_drv["Retained Trucks"]
    fleet_melt = fleet_drv.melt(
        id_vars="Date",
        value_vars=["Techs", "Sales Staff", "Trucks", "Retained Trucks", "Total Fleet Units"],
        var_name="Series",
        value_name="Value",
    )
    fleet_fig = px.line(
        fleet_melt,
        x="Date",
        y="Value",
        color="Series",
        line_dash="Series",
        line_dash_map={
            "Techs": "solid",
            "Sales Staff": "dot",
            "Trucks": "dash",
            "Retained Trucks": "dashdot",
            "Total Fleet Units": "longdash",
        },
        title="Staffing and Fleet Drivers",
    )
    st.plotly_chart(fleet_fig, width="stretch")
    if len(fleet_drv) and (fleet_drv["Trucks"] - fleet_drv["Techs"]).abs().max() < 1e-9:
        st.caption("`Trucks` overlaps `Techs` because `trucks_per_tech` is effectively 1.0 in this view range.")
    st.plotly_chart(
        px.line(
            range_df,
            x="Date",
            y=["Service Revenue", "Replacement Revenue", "Maintenance Revenue", "Upsell Revenue", "New Build Revenue"],
            title="Revenue Drivers",
        ),
        width="stretch",
    )

    st.subheader("Input Time Series Validation")
    input_scope = st.radio(
        "Input time-series scope",
        options=["Selected range", "Full horizon"],
        horizontal=True,
        key="input_ts_scope",
    )
    input_dates = pd.DatetimeIndex(range_df["Date"]) if input_scope == "Selected range" else pd.DatetimeIndex(view_df["Date"])
    input_ts_df = _build_input_timeseries(assumptions, input_dates)

    input_numeric_cols = [c for c in input_ts_df.columns if c != "Date" and pd.api.types.is_numeric_dtype(input_ts_df[c])]
    default_input_ts_cols = [
        c
        for c in [
            "calls_per_tech_per_day",
            "trucks_per_tech",
            "repl_close_rate",
            "avg_service_ticket",
            "tech_staffing_events_hires_input",
            "tech_staffing_events_attrition_input",
            "sales_staffing_events_hires_input",
            "sales_staffing_events_attrition_input",
            "res_new_build_install_schedule_installs_input",
            "lc_new_build_install_schedule_installs_input",
        ]
        if c in input_numeric_cols
    ]
    if "input_ts_selected_cols" not in st.session_state:
        st.session_state["input_ts_selected_cols"] = default_input_ts_cols
    else:
        st.session_state["input_ts_selected_cols"] = [
            c for c in st.session_state.get("input_ts_selected_cols", []) if c in input_numeric_cols
        ]
    st.multiselect(
        "Chart input series",
        options=input_numeric_cols,
        key="input_ts_selected_cols",
    )
    if st.session_state["input_ts_selected_cols"]:
        st.plotly_chart(
            px.line(
                input_ts_df,
                x="Date",
                y=st.session_state["input_ts_selected_cols"],
                title=f"Input Time Series ({input_scope})",
            ),
            width="stretch",
        )
    else:
        st.caption("Select one or more input series to plot.")

    with st.expander("All Input Time Series Table", expanded=False):
        st.dataframe(_format_dataframe_for_display(input_ts_df), width="stretch")
        st.download_button(
            f"Download Input Time Series CSV ({input_scope})",
            input_ts_df.to_csv(index=False),
            file_name=f"hvac_input_timeseries_{'range' if input_scope == 'Selected range' else 'full'}.csv",
            mime="text/csv",
        )

    st.subheader("Custom Visualizations (up to 4)")
    numeric_cols = [c for c in range_df.columns if pd.api.types.is_numeric_dtype(range_df[c])]
    for slot in range(1, 5):
        with st.expander(f"Custom Chart {slot}", expanded=False):
            st.checkbox("Enable", key=f"chart_{slot}_enabled")
            st.text_input("Title", key=f"chart_{slot}_title")
            st.selectbox("Chart Type", ["line", "area", "bar"], key=f"chart_{slot}_type")
            st.multiselect("Y Columns", numeric_cols, key=f"chart_{slot}_cols")
            if st.session_state[f"chart_{slot}_enabled"] and st.session_state[f"chart_{slot}_cols"]:
                cols = st.session_state[f"chart_{slot}_cols"]
                title = st.session_state[f"chart_{slot}_title"] or f"Custom Chart {slot}"
                ctype = st.session_state[f"chart_{slot}_type"]
                if ctype == "line":
                    fig = px.line(range_df, x="Date", y=cols, title=title)
                elif ctype == "area":
                    melt = range_df[["Date"] + cols].melt("Date", var_name="Series", value_name="Value")
                    fig = px.area(melt, x="Date", y="Value", color="Series", title=title)
                else:
                    melt = range_df[["Date"] + cols].melt("Date", var_name="Series", value_name="Value")
                    fig = px.bar(melt, x="Date", y="Value", color="Series", barmode="group", title=title)
                st.plotly_chart(fig, width="stretch")

with sens_tab:
    st.subheader("One-way sensitivity")
    candidate_drivers = available_sensitivity_drivers(assumptions)
    default_sens_drivers = [d for d in DEFAULT_SENSITIVITY_DRIVERS if d in candidate_drivers]
    if not st.session_state.get("sensitivity_drivers"):
        st.session_state["sensitivity_drivers"] = default_sens_drivers
    st.multiselect("Sensitivity Drivers", candidate_drivers, key="sensitivity_drivers")
    st.slider("Sensitivity Delta Percent", min_value=0.01, max_value=0.5, step=0.01, key="sensitivity_delta")
    st.toggle(
        "Auto-refresh sensitivity",
        key="sensitivity_auto_refresh",
        help="When off, sensitivity only recalculates when you click Run / Refresh.",
    )
    run_sensitivity = st.button(
        "Run / Refresh Sensitivity",
        help="Recompute one-way sensitivity results using the current assumptions, selected drivers, and delta.",
    )
    sensitivity_signature = (
        _serialize_assumptions(assumptions),
        float(st.session_state["sensitivity_delta"]),
        tuple(sorted(st.session_state["sensitivity_drivers"])),
    )
    should_refresh = (
        st.session_state["sensitivity_auto_refresh"]
        or run_sensitivity
        or st.session_state.get("sensitivity_result_signature") != sensitivity_signature
        and st.session_state.get("sensitivity_result_df") is None
    )
    if should_refresh and st.session_state["sensitivity_drivers"]:
        sens_df, target_year = _run_sensitivity_cached(
            sensitivity_signature[0],
            sensitivity_signature[1],
            sensitivity_signature[2],
        )
        st.session_state["sensitivity_result_df"] = sens_df
        st.session_state["sensitivity_result_year"] = target_year
        st.session_state["sensitivity_result_signature"] = sensitivity_signature
    elif not st.session_state["sensitivity_auto_refresh"] and st.session_state.get("sensitivity_result_signature") != sensitivity_signature:
        st.info("Sensitivity inputs changed. Click `Run / Refresh Sensitivity` to update results.")

    sens_df = st.session_state.get("sensitivity_result_df", pd.DataFrame())
    target_year = st.session_state.get("sensitivity_result_year", 1)
    target = st.selectbox("Target metric", TARGET_OPTIONS, index=TARGET_OPTIONS.index("Year N EBITDA"))
    if len(sens_df) > 0:
        tornado = sens_df.pivot(index="Driver", columns="Case", values=f"Delta {target}").fillna(0)
        tornado["Base"] = 0
        tdf = tornado[["Low", "Base", "High"]].reset_index().melt(id_vars="Driver", var_name="Case", value_name="Delta")
        st.plotly_chart(
            px.bar(
                tdf,
                x="Delta",
                y="Driver",
                color="Case",
                orientation="h",
                title=f"Tornado Chart for {target} (Year N = {target_year})",
            ),
            width="stretch",
        )
        st.dataframe(_format_dataframe_for_display(sens_df), width="stretch")

        insight_df = _build_sensitivity_insights(sens_df, target, float(st.session_state["sensitivity_delta"]))
        if len(insight_df) > 0:
            st.subheader("Value Leakage vs Value Creation Insights")
            direction_note = "For this target, higher is better." if _sensitivity_objective_sign(target) > 0 else "For this target, lower is better; reductions are treated as value gain."
            st.caption(
                f"Interpretation based on +/-{100 * float(st.session_state['sensitivity_delta']):.1f}% driver shocks. "
                + direction_note
            )

            leakage_df = insight_df.sort_values(["Leakage Risk", "Sensitivity (abs)"], ascending=[False, False]).head(10)
            value_df = insight_df.sort_values(["Value Gain Potential", "Leakage Risk"], ascending=[False, True]).head(10)
            leakage_total_top5 = float(leakage_df["Leakage Risk"].head(5).sum())
            value_total_top5 = float(value_df["Value Gain Potential"].head(5).sum())

            k1, k2, k3 = st.columns(3)
            k1.metric("Top-5 Leakage Exposure", _format_money_like(leakage_total_top5))
            k2.metric("Top-5 Value Gain Potential", _format_money_like(value_total_top5))
            k3.metric("Gain-to-Leakage Ratio", f"{(value_total_top5 / max(leakage_total_top5, 1e-9)):,.2f}x")

            st.markdown("Leakage Priorities (defend value)")
            st.dataframe(
                _format_dataframe_for_display(
                    leakage_df[
                        [
                            "Driver",
                            "Leakage Risk",
                            "Net Opportunity",
                            "Focus Direction",
                            "Sensitivity per 1% Input Move",
                            "Posture",
                        ]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

            st.markdown("Value Creation Priorities (grow value)")
            st.dataframe(
                _format_dataframe_for_display(
                    value_df[
                        [
                            "Driver",
                            "Value Gain Potential",
                            "Focus Direction",
                            "Focus Delta (Target)",
                            "Sensitivity per 1% Input Move",
                            "Posture",
                        ]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

            with st.expander("Full Driver Insight Detail", expanded=False):
                st.dataframe(_format_dataframe_for_display(insight_df), width="stretch", hide_index=True)
    else:
        st.info("No sensitivity drivers selected.")
