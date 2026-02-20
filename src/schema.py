"""Scenario schema helpers, constants, and migration utilities."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

import pandas as pd

from src.defaults import DEFAULTS


SCHEMA_VERSION = 2
SCENARIO_TYPE = "scenario"
WORKSPACE_TYPE = "workspace"

VALUE_MODES = {"nominal", "real_inflation", "real_pv"}
NEW_BUILD_MODES = {"schedule", "base_seasonal", "annual_total"}
ASSET_EXPIRY_MODES = {"release", "retain", "salvage"}

MONTH_NAME_BY_NUM = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
MONTH_NUM_BY_NAME = {name: num for num, name in MONTH_NAME_BY_NUM.items()}

DEPRECATED_V1_FIELDS = {
    "tech_hire_per_quarter",
    "work_days_per_month",
    "avg_hours_per_tech_per_month",
}


def month_name_option(month_num: int) -> str:
    return f"{MONTH_NAME_BY_NUM[int(month_num)]} ({int(month_num)})"


def parse_month_name_option(option: str) -> int:
    if "(" in option and option.endswith(")"):
        try:
            return int(option.split("(")[-1].rstrip(")"))
        except ValueError:
            pass
    label = option.strip()
    if label in MONTH_NUM_BY_NAME:
        return MONTH_NUM_BY_NAME[label]
    raise ValueError(f"Unrecognized month option: {option}")


def month_label(year: int, month: int) -> str:
    return f"{int(year):04d}-{int(month):02d}"


def _within_horizon(month_str: str, start_date: pd.Timestamp, horizon_months: int) -> bool:
    try:
        d = pd.Timestamp(f"{month_str}-01")
    except Exception:
        return False
    end_date = start_date + pd.offsets.MonthBegin(horizon_months)
    return start_date <= d < end_date


def _sanitize_event_list(
    raw_events: Any, start_date: pd.Timestamp, horizon_months: int, warnings: list[str], key_name: str
) -> list[dict]:
    sanitized: list[dict] = []
    if raw_events is None:
        return sanitized
    if isinstance(raw_events, pd.DataFrame):
        records = raw_events.to_dict(orient="records")
    elif isinstance(raw_events, list):
        records = raw_events
    else:
        warnings.append(f"{key_name} ignored because it is not a list/table.")
        return sanitized

    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            warnings.append(f"{key_name}[{idx}] ignored because entry is not an object.")
            continue
        month = str(item.get("month", "")).strip()
        if len(month) != 7 or month[4] != "-":
            warnings.append(f"{key_name}[{idx}] ignored due to invalid month format.")
            continue
        if not _within_horizon(month, start_date, horizon_months):
            warnings.append(f"{key_name}[{idx}] ignored because month is outside the forecast horizon.")
            continue
        try:
            hires = max(0, int(item.get("hires", 0)))
            attrition = max(0, int(item.get("attrition", 0)))
        except (TypeError, ValueError):
            warnings.append(f"{key_name}[{idx}] ignored because hires/attrition is invalid.")
            continue
        sanitized.append({"month": month, "hires": hires, "attrition": attrition})
    return sanitized


def _sanitize_schedule_list(
    raw_schedule: Any, start_date: pd.Timestamp, horizon_months: int, warnings: list[str], key_name: str
) -> list[dict]:
    sanitized: list[dict] = []
    if raw_schedule is None:
        return sanitized
    if isinstance(raw_schedule, pd.DataFrame):
        records = raw_schedule.to_dict(orient="records")
    elif isinstance(raw_schedule, list):
        records = raw_schedule
    else:
        warnings.append(f"{key_name} ignored because it is not a list/table.")
        return sanitized

    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            warnings.append(f"{key_name}[{idx}] ignored because entry is not an object.")
            continue
        month = str(item.get("month", "")).strip()
        if len(month) != 7 or month[4] != "-":
            warnings.append(f"{key_name}[{idx}] ignored due to invalid month format.")
            continue
        if not _within_horizon(month, start_date, horizon_months):
            warnings.append(f"{key_name}[{idx}] ignored because month is outside the forecast horizon.")
            continue
        try:
            installs = max(0.0, float(item.get("installs", 0.0)))
        except (TypeError, ValueError):
            warnings.append(f"{key_name}[{idx}] ignored because installs value is invalid.")
            continue
        sanitized.append({"month": month, "installs": installs})
    return sanitized


def _legacy_quarterly_events(inputs: dict, legacy_hires_per_quarter: Any) -> list[dict]:
    try:
        hires_per_quarter = int(max(0, legacy_hires_per_quarter))
    except (TypeError, ValueError):
        hires_per_quarter = 0
    if hires_per_quarter == 0:
        return []
    start_date = datetime(int(inputs["start_year"]), int(inputs["start_month"]), 1)
    horizon = int(inputs["horizon_months"])
    events = []
    for m in range(3, horizon, 3):
        d = pd.Timestamp(start_date) + pd.offsets.MonthBegin(m)
        events.append({"month": d.strftime("%Y-%m"), "hires": hires_per_quarter, "attrition": 0})
    return events


def migrate_assumptions(raw_inputs: dict) -> tuple[dict, list[str], list[str]]:
    """Migrate incoming assumptions into v2 schema."""
    warnings: list[str] = []
    unknown_keys: list[str] = []
    legacy_fields: dict[str, Any] = {}
    inputs = deepcopy(DEFAULTS)
    payload = raw_inputs if isinstance(raw_inputs, dict) else {}

    for k, v in payload.items():
        if k in inputs:
            inputs[k] = v
        elif k in DEPRECATED_V1_FIELDS:
            legacy_fields[k] = v
        else:
            unknown_keys.append(k)

    for field in DEPRECATED_V1_FIELDS:
        if field in legacy_fields:
            warnings.append(f"{field} is deprecated and was migrated/ignored in schema v2.")

    # Bridge old maintenance keys if provided from v1 payloads.
    if "agreements_start" in payload and "res_agreements_start" not in payload:
        inputs["res_agreements_start"] = payload["agreements_start"]
    if "new_agreements_per_month" in payload and "res_new_agreements_per_month" not in payload:
        inputs["res_new_agreements_per_month"] = payload["new_agreements_per_month"]
    if "churn_annual_pct" in payload and "res_churn_annual_pct" not in payload:
        inputs["res_churn_annual_pct"] = payload["churn_annual_pct"]
    if "maint_monthly_fee" in payload and "res_maint_monthly_fee" not in payload:
        inputs["res_maint_monthly_fee"] = payload["maint_monthly_fee"]
    if "cost_per_maint_visit" in payload and "res_cost_per_maint_visit" not in payload:
        inputs["res_cost_per_maint_visit"] = payload["cost_per_maint_visit"]
    if "maint_visits_per_agreement_per_year" in payload and "res_maint_visits_per_agreement_per_year" not in payload:
        inputs["res_maint_visits_per_agreement_per_year"] = payload["maint_visits_per_agreement_per_year"]

    start_date = pd.Timestamp(datetime(int(inputs["start_year"]), int(inputs["start_month"]), 1))
    horizon_months = int(inputs["horizon_months"])

    if not inputs.get("tech_staffing_events"):
        legacy_events = _legacy_quarterly_events(inputs, legacy_fields.get("tech_hire_per_quarter"))
        if legacy_events:
            inputs["tech_staffing_events"] = legacy_events
            warnings.append("Migrated legacy tech_hire_per_quarter to tech_staffing_events.")

    inputs["tech_staffing_events"] = _sanitize_event_list(
        inputs.get("tech_staffing_events"), start_date, horizon_months, warnings, "tech_staffing_events"
    )
    inputs["sales_staffing_events"] = _sanitize_event_list(
        inputs.get("sales_staffing_events"), start_date, horizon_months, warnings, "sales_staffing_events"
    )
    inputs["res_new_build_install_schedule"] = _sanitize_schedule_list(
        inputs.get("res_new_build_install_schedule"), start_date, horizon_months, warnings, "res_new_build_install_schedule"
    )
    inputs["lc_new_build_install_schedule"] = _sanitize_schedule_list(
        inputs.get("lc_new_build_install_schedule"), start_date, horizon_months, warnings, "lc_new_build_install_schedule"
    )

    # Enumerations and clamping.
    inputs["new_build_mode"] = str(inputs.get("new_build_mode", "base_seasonal"))
    if inputs["new_build_mode"] not in NEW_BUILD_MODES:
        warnings.append("new_build_mode invalid; reset to base_seasonal.")
        inputs["new_build_mode"] = "base_seasonal"

    inputs["asset_expiry_mode"] = str(inputs.get("asset_expiry_mode", "release"))
    if inputs["asset_expiry_mode"] not in ASSET_EXPIRY_MODES:
        warnings.append("asset_expiry_mode invalid; reset to release.")
        inputs["asset_expiry_mode"] = "release"

    inputs["value_mode"] = str(inputs.get("value_mode", "nominal"))
    if inputs["value_mode"] not in VALUE_MODES:
        warnings.append("value_mode invalid; reset to nominal.")
        inputs["value_mode"] = "nominal"

    # Scalar clamping.
    inputs["start_year"] = int(min(2100, max(2000, int(inputs["start_year"]))))
    inputs["start_month"] = int(min(12, max(1, int(inputs["start_month"]))))
    inputs["horizon_months"] = int(min(120, max(12, int(inputs["horizon_months"]))))
    inputs["peak_month"] = int(min(12, max(1, int(inputs["peak_month"]))))
    inputs["raise_effective_month"] = int(min(12, max(1, int(inputs["raise_effective_month"]))))
    for year_key in ("manager_start_year", "ops_manager_start_year", "marketing_manager_start_year"):
        inputs[year_key] = int(min(2100, max(2000, int(inputs[year_key]))))
    for month_key in ("manager_start_month", "ops_manager_start_month", "marketing_manager_start_month"):
        inputs[month_key] = int(min(12, max(1, int(inputs[month_key]))))

    pct_like = [
        "monthly_price_growth",
        "monthly_cost_inflation",
        "seasonality_amplitude",
        "payroll_burden_pct",
        "service_material_pct",
        "attach_rate",
        "repl_close_rate",
        "repl_equipment_pct",
        "financing_penetration",
        "financing_fee_pct",
        "truck_downpayment_pct",
        "truck_financed_pct",
        "loan_annual_rate",
        "loc_annual_rate",
        "distributions_pct_of_ebitda",
        "sales_repl_close_lift_per_fte",
        "sales_repl_close_rate_cap",
        "res_capacity_pct",
        "asset_salvage_pct",
        "res_maint_hybrid_weight_calls",
        "lc_maint_hybrid_weight_calls",
        "res_service_upsell_conversion_pct",
        "lc_service_upsell_conversion_pct",
        "res_maint_upsell_conversion_pct",
        "lc_maint_upsell_conversion_pct",
        "res_service_upsell_gross_margin_pct",
        "lc_service_upsell_gross_margin_pct",
        "res_maint_upsell_gross_margin_pct",
        "lc_maint_upsell_gross_margin_pct",
        "res_new_build_gross_margin_pct",
        "lc_new_build_gross_margin_pct",
        "annual_raise_pct_tech",
        "annual_raise_pct_sales",
        "annual_raise_pct_manager",
        "annual_raise_pct_ops_manager",
        "annual_raise_pct_marketing_manager",
    ]
    for key in pct_like:
        try:
            inputs[key] = float(min(1.0, max(0.0, float(inputs[key]))))
        except (TypeError, ValueError):
            inputs[key] = float(DEFAULTS[key])
            warnings.append(f"{key} invalid and reset to default.")

    bool_keys = [k for k, v in DEFAULTS.items() if isinstance(v, bool)]
    for key in bool_keys:
        val = inputs.get(key, DEFAULTS[key])
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            inputs[key] = bool(val)
            continue
        if isinstance(val, str):
            txt = val.strip().lower()
            if txt in {"1", "true", "yes", "y", "on"}:
                inputs[key] = True
                continue
            if txt in {"0", "false", "no", "n", "off"}:
                inputs[key] = False
                continue
        inputs[key] = bool(DEFAULTS[key])
        warnings.append(f"{key} invalid and reset to default.")

    non_negative = [
        k
        for k, v in DEFAULTS.items()
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and k not in {"start_year", "start_month", "horizon_months", "peak_month", "raise_effective_month"}
    ]
    for key in non_negative:
        try:
            if isinstance(DEFAULTS[key], int):
                inputs[key] = max(0, int(inputs[key]))
            else:
                inputs[key] = max(0.0, float(inputs[key]))
        except (TypeError, ValueError):
            inputs[key] = deepcopy(DEFAULTS[key])
            warnings.append(f"{key} invalid and reset to default.")

    return inputs, warnings, sorted(unknown_keys)


def migrate_import_payload(payload: dict) -> tuple[dict, dict | None, list[str], list[str]]:
    """Parse imported scenario/workspace payload and return migrated assumptions + ui_state."""
    if not isinstance(payload, dict):
        return deepcopy(DEFAULTS), None, ["Import payload is not a JSON object."], []

    payload_type = payload.get("type")
    if payload_type in {SCENARIO_TYPE, WORKSPACE_TYPE}:
        assumptions_raw = payload.get("assumptions", {})
        ui_state = payload.get("ui_state") if payload_type == WORKSPACE_TYPE else None
        assumptions, warnings, unknown = migrate_assumptions(assumptions_raw)
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            warnings.append(f"Imported schema_version={version}; migrated to schema_version={SCHEMA_VERSION}.")
        return assumptions, ui_state, warnings, unknown

    assumptions, warnings, unknown = migrate_assumptions(payload)
    warnings.append("Imported legacy assumption JSON without bundle metadata.")
    return assumptions, None, warnings, unknown
