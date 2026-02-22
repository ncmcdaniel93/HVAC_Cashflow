"""Core HVAC cash flow model engine (schema v2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from src.calendar_utils import derive_workdays_and_hours
from src.schema import ASSET_EXPIRY_MODES, NEW_BUILD_MODES


@dataclass
class ModelInputs:
    data: Dict

    def __getattr__(self, key):
        if key in self.data:
            return self.data[key]
        raise AttributeError(key)


def _start_mask(dates: pd.DatetimeIndex, year: int, month: int) -> np.ndarray:
    start = pd.Timestamp(datetime(int(year), int(month), 1))
    return (dates >= start).astype(float)


def _raise_factor(dates: pd.DatetimeIndex, raise_month: int, annual_raise_pct: float) -> np.ndarray:
    factor = np.ones(len(dates), dtype=float)
    raises_applied = 0
    for idx, d in enumerate(dates):
        if idx > 0 and d.month == int(raise_month):
            raises_applied += 1
        factor[idx] = (1 + annual_raise_pct) ** raises_applied
    return factor


def _term_payment(principal: float, annual_rate: float, term_months: int) -> float:
    if principal <= 0 or term_months <= 0:
        return 0.0
    r = annual_rate / 12
    if r == 0:
        return principal / term_months
    return principal * r / (1 - (1 + r) ** (-term_months))


def _event_series(events: list[dict], dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    idx_map = {d.strftime("%Y-%m"): m for m, d in enumerate(dates)}
    hires = np.zeros(len(dates), dtype=float)
    attrition = np.zeros(len(dates), dtype=float)
    for event in events:
        month = str(event.get("month", ""))
        if month not in idx_map:
            continue
        m = idx_map[month]
        hires[m] += max(0.0, float(event.get("hires", 0.0)))
        attrition[m] += max(0.0, float(event.get("attrition", 0.0)))
    return hires, attrition


def _schedule_series(schedule: list[dict], dates: pd.DatetimeIndex) -> np.ndarray:
    idx_map = {d.strftime("%Y-%m"): m for m, d in enumerate(dates)}
    installs = np.zeros(len(dates), dtype=float)
    for row in schedule:
        month = str(row.get("month", ""))
        if month in idx_map:
            installs[idx_map[month]] += max(0.0, float(row.get("installs", 0.0)))
    return installs


def _quarterly_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    quarter_months = {1, 4, 7, 10}
    start = pd.Timestamp(dates[0])
    start_year = int(start.year)
    start_month = int(start.month)

    # Calendar-quarter anchoring with first event at next quarter boundary when start is mid-quarter.
    next_q_month = None
    for m in sorted(quarter_months):
        if m > start_month:
            next_q_month = m
            break
    boundary_year = start_year
    if start_month in quarter_months:
        next_q_month = start_month
    elif next_q_month is None:
        next_q_month = 1
        boundary_year += 1
    first_boundary = pd.Timestamp(datetime(boundary_year, next_q_month, 1))
    return np.array([(d >= first_boundary and d.month in quarter_months) for d in dates], dtype=bool)


def _new_build_installs(
    mode: str,
    dates: pd.DatetimeIndex,
    schedule: list[dict],
    base_monthly: float,
    annual_total: float,
    seasonality_mult: np.ndarray,
) -> np.ndarray:
    if mode == "schedule":
        return _schedule_series(schedule, dates)
    if mode == "annual_total":
        return np.full(len(dates), float(annual_total) / 12.0)
    return np.maximum(0.0, float(base_monthly) * seasonality_mult)


def _seasonality_multiplier(month_numbers: np.ndarray, amplitude: float, peak_month: int) -> np.ndarray:
    """Return seasonality multiplier where the maximum occurs at peak_month."""
    # Cosine reaches its maximum (1.0) at zero phase difference.
    return 1 + float(amplitude) * np.cos(2 * np.pi * (month_numbers - int(peak_month)) / 12.0)


def validate_inputs(inputs: Dict) -> None:
    pct_fields = [
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
    for field in pct_fields:
        val = float(inputs[field])
        if not (0 <= val <= 1):
            raise ValueError(f"{field} must be in [0,1].")

    non_negative_fields = [
        "horizon_months",
        "start_year",
        "start_month",
        "peak_month",
        "raise_effective_month",
        "starting_techs",
        "sales_starting_staff",
        "max_techs",
        "calls_per_tech_per_day",
        "tech_hours_per_day",
        "sales_hours_per_day",
        "tech_wage_per_hour",
        "sales_wage_per_hour",
        "tools_per_new_tech_capex",
        "asset_reuse_lag_months",
        "avg_service_ticket",
        "repl_leads_per_tech_per_month",
        "avg_repl_ticket",
        "permit_cost_per_repl_job",
        "disposal_cost_per_repl_job",
        "res_agreements_start",
        "res_new_agreements_per_month",
        "res_maint_monthly_fee",
        "res_cost_per_maint_visit",
        "res_maint_visits_per_agreement_per_year",
        "lc_agreements_start",
        "lc_new_agreements_per_month",
        "lc_maint_quarterly_fee",
        "lc_cost_per_maint_visit",
        "lc_maint_visits_per_agreement_per_year",
        "maint_visits_capacity_per_tech_per_month",
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
        "office_payroll_monthly",
        "rent_monthly",
        "utilities_monthly",
        "insurance_monthly",
        "software_monthly",
        "other_fixed_monthly",
        "manager_salary_monthly",
        "ops_manager_salary_monthly",
        "marketing_manager_salary_monthly",
        "ar_days",
        "ap_days",
        "inventory_days",
        "opening_ar_balance",
        "opening_ap_balance",
        "opening_inventory_balance",
        "starting_cash",
        "loan_principal",
        "loan_term_months",
        "loc_limit",
        "min_cash_target",
        "loc_repay_buffer",
        "distributions_only_if_cash_above",
        "discount_rate_annual_nominal",
    ]
    for field in non_negative_fields:
        if float(inputs[field]) < 0:
            raise ValueError(f"{field} must be non-negative.")

    if not (1 <= int(inputs["start_month"]) <= 12):
        raise ValueError("start_month must be in [1,12].")
    if not (1 <= int(inputs["peak_month"]) <= 12):
        raise ValueError("peak_month must be in [1,12].")
    if not (1 <= int(inputs["raise_effective_month"]) <= 12):
        raise ValueError("raise_effective_month must be in [1,12].")
    if int(inputs["horizon_months"]) < 1:
        raise ValueError("horizon_months must be at least 1.")
    if int(inputs["loan_term_months"]) < 1:
        raise ValueError("loan_term_months must be at least 1.")
    if int(inputs["max_techs"]) < int(inputs["starting_techs"]):
        raise ValueError("max_techs must be >= starting_techs.")
    if inputs["paid_leads_mode"] not in {"fixed", "per_tech"}:
        raise ValueError("paid_leads_mode must be 'fixed' or 'per_tech'.")
    if inputs["capex_trucks_mode"] not in {"payments_only", "purchase_with_downpayment"}:
        raise ValueError("capex_trucks_mode must be 'payments_only' or 'purchase_with_downpayment'.")
    if inputs["new_build_mode"] not in NEW_BUILD_MODES:
        raise ValueError("new_build_mode must be one of schedule/base_seasonal/annual_total.")
    if inputs["asset_expiry_mode"] not in ASSET_EXPIRY_MODES:
        raise ValueError("asset_expiry_mode must be one of release/retain/salvage.")


def run_model(raw_inputs: Dict) -> pd.DataFrame:
    validate_inputs(raw_inputs)
    i = ModelInputs(raw_inputs)

    start_date = datetime(int(i.start_year), int(i.start_month), 1)
    dates = pd.date_range(start=start_date, periods=int(i.horizon_months), freq="MS")
    horizon = len(dates)
    t = np.arange(horizon, dtype=float)
    months = dates.month.values

    seasonality = _seasonality_multiplier(months, i.seasonality_amplitude, i.peak_month)
    new_build_seasonality = _seasonality_multiplier(months, i.new_build_seasonality_amplitude, i.peak_month)
    price_mult = (1 + i.monthly_price_growth) ** t
    cost_mult = (1 + i.monthly_cost_inflation) ** t

    tech_hires_raw, tech_attrition_raw = _event_series(i.tech_staffing_events, dates)
    sales_hires_raw, sales_attrition_raw = _event_series(i.sales_staffing_events, dates)

    techs = np.zeros(horizon)
    sales_staff = np.zeros(horizon)
    new_techs = np.zeros(horizon)
    tech_attrition = np.zeros(horizon)
    new_sales = np.zeros(horizon)
    sales_attrition = np.zeros(horizon)
    tech_prev = float(i.starting_techs)
    sales_prev = float(i.sales_starting_staff)
    for m in range(horizon):
        # Process gross attrition and gross hires separately so same-month churn + hiring
        # remains visible in outputs and capex/asset-reuse math.
        tech_attr_eff = min(max(0.0, tech_attrition_raw[m]), tech_prev)
        tech_after_attr = tech_prev - tech_attr_eff
        tech_hire_eff = min(max(0.0, tech_hires_raw[m]), max(0.0, float(i.max_techs) - tech_after_attr))
        tech_curr = tech_after_attr + tech_hire_eff

        sales_attr_eff = min(max(0.0, sales_attrition_raw[m]), sales_prev)
        sales_after_attr = sales_prev - sales_attr_eff
        sales_hire_eff = max(0.0, sales_hires_raw[m])
        sales_curr = sales_after_attr + sales_hire_eff

        techs[m] = tech_curr
        sales_staff[m] = sales_curr
        new_techs[m] = tech_hire_eff
        tech_attrition[m] = tech_attr_eff
        new_sales[m] = sales_hire_eff
        sales_attrition[m] = sales_attr_eff
        tech_prev = tech_curr
        sales_prev = sales_curr

    wd = derive_workdays_and_hours(
        dates=dates,
        techs=techs,
        sales_staff=sales_staff,
        tech_hours_per_day=float(i.tech_hours_per_day),
        sales_hours_per_day=float(i.sales_hours_per_day),
    )
    workdays = wd.workdays
    tech_hours = wd.tech_hours
    sales_hours = wd.sales_hours

    tech_raise = _raise_factor(dates, int(i.raise_effective_month), float(i.annual_raise_pct_tech))
    sales_raise = _raise_factor(dates, int(i.raise_effective_month), float(i.annual_raise_pct_sales))
    mgr_raise = _raise_factor(dates, int(i.raise_effective_month), float(i.annual_raise_pct_manager))
    ops_raise = _raise_factor(dates, int(i.raise_effective_month), float(i.annual_raise_pct_ops_manager))
    mkt_raise = _raise_factor(dates, int(i.raise_effective_month), float(i.annual_raise_pct_marketing_manager))

    calls = techs * i.calls_per_tech_per_day * workdays
    res_calls = calls * i.res_capacity_pct
    lc_calls = calls - res_calls
    service_rev = calls * i.avg_service_ticket * seasonality * price_mult

    repl_leads = techs * i.repl_leads_per_tech_per_month
    repl_close_rate = np.minimum(i.sales_repl_close_rate_cap, i.repl_close_rate + sales_staff * i.sales_repl_close_lift_per_fte)
    repl_jobs = repl_leads * repl_close_rate
    repl_rev = repl_jobs * i.avg_repl_ticket * seasonality * price_mult

    res_maint_agreements = np.zeros(horizon)
    lc_maint_agreements = np.zeros(horizon)
    res_maint_rev = np.zeros(horizon)
    lc_maint_rev = np.zeros(horizon)
    res_maint_visits = np.zeros(horizon)
    lc_maint_visits = np.zeros(horizon)
    res_maint_direct_cost = np.zeros(horizon)
    lc_maint_direct_cost = np.zeros(horizon)
    res_new_agreements = np.zeros(horizon)
    lc_new_agreements = np.zeros(horizon)

    if i.enable_maintenance:
        res_balance = float(i.res_agreements_start)
        lc_balance = float(i.lc_agreements_start)
        res_retention = (1 - i.res_churn_annual_pct) ** (1 / 12)
        lc_retention = (1 - i.lc_churn_annual_pct) ** (1 / 12)
        lc_quarterly = _quarterly_mask(dates)

        for m in range(horizon):
            res_call_gen = res_calls[m] * i.res_maint_call_conversion_pct
            res_tech_gen = techs[m] * i.res_maint_agreements_per_tech_per_month
            res_new = i.res_maint_hybrid_weight_calls * res_call_gen + (1 - i.res_maint_hybrid_weight_calls) * res_tech_gen
            res_new += i.res_new_agreements_per_month

            lc_call_gen = lc_calls[m] * i.lc_maint_call_conversion_pct
            lc_tech_gen = techs[m] * i.lc_maint_agreements_per_tech_per_month
            lc_new = i.lc_maint_hybrid_weight_calls * lc_call_gen + (1 - i.lc_maint_hybrid_weight_calls) * lc_tech_gen
            lc_new += i.lc_new_agreements_per_month

            res_balance = max(0.0, res_balance * res_retention + res_new)
            lc_balance = max(0.0, lc_balance * lc_retention + lc_new)

            if i.maint_visits_capacity_per_tech_per_month > 0:
                res_cap = (
                    techs[m] * i.maint_visits_capacity_per_tech_per_month * 12 / max(i.res_maint_visits_per_agreement_per_year, 1e-9)
                )
                lc_cap = (
                    techs[m] * i.maint_visits_capacity_per_tech_per_month * 12 / max(i.lc_maint_visits_per_agreement_per_year, 1e-9)
                )
                res_balance = min(res_balance, res_cap)
                lc_balance = min(lc_balance, lc_cap)

            res_maint_agreements[m] = res_balance
            lc_maint_agreements[m] = lc_balance
            res_new_agreements[m] = max(0.0, res_new)
            lc_new_agreements[m] = max(0.0, lc_new)

            res_maint_visits[m] = res_balance * (i.res_maint_visits_per_agreement_per_year / 12)
            if lc_quarterly[m]:
                lc_maint_visits[m] = lc_balance * (i.lc_maint_visits_per_agreement_per_year / 4)
                lc_maint_rev[m] = lc_balance * i.lc_maint_quarterly_fee * price_mult[m]
            else:
                lc_maint_visits[m] = 0.0
                lc_maint_rev[m] = 0.0

            res_maint_rev[m] = res_balance * i.res_maint_monthly_fee * price_mult[m]
            res_maint_direct_cost[m] = res_maint_visits[m] * i.res_cost_per_maint_visit * cost_mult[m]
            lc_maint_direct_cost[m] = lc_maint_visits[m] * i.lc_cost_per_maint_visit * cost_mult[m]

    maint_agreements = res_maint_agreements + lc_maint_agreements
    maint_rev = res_maint_rev + lc_maint_rev
    maint_direct_cost = res_maint_direct_cost + lc_maint_direct_cost

    # Upsell economics by segment and visit type (incremental).
    res_service_upsell_rev = (
        res_calls * i.res_service_upsell_conversion_pct * i.res_service_upsell_revenue_per_visit * price_mult
    )
    lc_service_upsell_rev = lc_calls * i.lc_service_upsell_conversion_pct * i.lc_service_upsell_revenue_per_visit * price_mult
    res_maint_upsell_rev = (
        res_maint_visits * i.res_maint_upsell_conversion_pct * i.res_maint_upsell_revenue_per_visit * price_mult
    )
    lc_maint_upsell_rev = (
        lc_maint_visits * i.lc_maint_upsell_conversion_pct * i.lc_maint_upsell_revenue_per_visit * price_mult
    )
    upsell_revenue = res_service_upsell_rev + lc_service_upsell_rev + res_maint_upsell_rev + lc_maint_upsell_rev
    upsell_direct_cost = (
        res_service_upsell_rev * (1 - i.res_service_upsell_gross_margin_pct)
        + lc_service_upsell_rev * (1 - i.lc_service_upsell_gross_margin_pct)
        + res_maint_upsell_rev * (1 - i.res_maint_upsell_gross_margin_pct)
        + lc_maint_upsell_rev * (1 - i.lc_maint_upsell_gross_margin_pct)
    )

    # New-build installs by segment.
    res_new_build_installs = _new_build_installs(
        mode=i.new_build_mode,
        dates=dates,
        schedule=i.res_new_build_install_schedule,
        base_monthly=i.res_new_build_installs_per_month,
        annual_total=i.res_new_build_annual_installs,
        seasonality_mult=new_build_seasonality,
    )
    lc_new_build_installs = _new_build_installs(
        mode=i.new_build_mode,
        dates=dates,
        schedule=i.lc_new_build_install_schedule,
        base_monthly=i.lc_new_build_installs_per_month,
        annual_total=i.lc_new_build_annual_installs,
        seasonality_mult=new_build_seasonality,
    )
    res_new_build_revenue = res_new_build_installs * i.res_new_build_avg_ticket * price_mult
    lc_new_build_revenue = lc_new_build_installs * i.lc_new_build_avg_ticket * price_mult
    res_new_build_direct_cost = res_new_build_revenue * (1 - i.res_new_build_gross_margin_pct)
    lc_new_build_direct_cost = lc_new_build_revenue * (1 - i.lc_new_build_gross_margin_pct)
    new_build_revenue = res_new_build_revenue + lc_new_build_revenue
    new_build_direct_cost = res_new_build_direct_cost + lc_new_build_direct_cost

    total_revenue = service_rev + repl_rev + maint_rev + upsell_revenue + new_build_revenue
    financing_fee_cost = repl_rev * i.financing_penetration * i.financing_fee_pct

    service_materials = service_rev * i.service_material_pct * cost_mult
    repl_equipment = repl_rev * i.repl_equipment_pct * cost_mult
    permits = repl_jobs * i.permit_cost_per_repl_job * cost_mult
    disposal = repl_jobs * i.disposal_cost_per_repl_job * cost_mult
    tech_labor_cost_per_wage_unit = tech_hours * (1 + i.payroll_burden_pct) * cost_mult * tech_raise
    direct_labor = tech_labor_cost_per_wage_unit * i.tech_wage_per_hour

    total_direct_costs = (
        service_materials
        + repl_equipment
        + permits
        + disposal
        + direct_labor
        + maint_direct_cost
        + financing_fee_cost
        + upsell_direct_cost
        + new_build_direct_cost
    )
    gross_profit = total_revenue - total_direct_costs

    fixed_opex = (
        i.office_payroll_monthly
        + i.rent_monthly
        + i.utilities_monthly
        + i.insurance_monthly
        + i.software_monthly
        + i.other_fixed_monthly
    ) * cost_mult

    paid_leads = (
        np.full(horizon, i.paid_leads_per_month, dtype=float)
        if i.paid_leads_mode == "fixed"
        else techs * i.paid_leads_per_tech_per_month
    )
    marketing_spend = paid_leads * i.cost_per_lead * cost_mult + i.branding_fixed_monthly * cost_mult

    # Asset reuse pools for tools and trucks.
    tools_capex = np.zeros(horizon)
    truck_capex = np.zeros(horizon)
    salvage_proceeds = np.zeros(horizon)
    retained_trucks = np.zeros(horizon)
    reused_tools = np.zeros(horizon)
    reused_trucks = np.zeros(horizon)

    lag = int(i.asset_reuse_lag_months)
    tool_pool: list[tuple[int, float]] = []
    truck_pool: list[tuple[int, float]] = []
    assigned_trucks = techs * i.trucks_per_tech

    for m in range(horizon):
        if i.asset_expiry_mode in {"release", "salvage"}:
            keep_tools: list[tuple[int, float]] = []
            keep_trucks: list[tuple[int, float]] = []
            for expiry, qty in tool_pool:
                if expiry < m:
                    if i.asset_expiry_mode == "salvage":
                        salvage_proceeds[m] += qty * i.tools_per_new_tech_capex * i.asset_salvage_pct
                else:
                    keep_tools.append((expiry, qty))
            for expiry, qty in truck_pool:
                if expiry < m:
                    if i.asset_expiry_mode == "salvage":
                        salvage_proceeds[m] += qty * i.truck_purchase_price * i.asset_salvage_pct
                else:
                    keep_trucks.append((expiry, qty))
            tool_pool = keep_tools
            truck_pool = keep_trucks

        if tech_attrition[m] > 0:
            expiry = m + lag
            tool_pool.append((expiry, tech_attrition[m]))
            truck_pool.append((expiry, tech_attrition[m] * i.trucks_per_tech))

        need_tools = new_techs[m]
        need_trucks = new_techs[m] * i.trucks_per_tech
        used_tools = 0.0
        used_trucks = 0.0

        if need_tools > 0 and tool_pool:
            updated_pool: list[tuple[int, float]] = []
            for expiry, qty in tool_pool:
                if need_tools <= 0:
                    updated_pool.append((expiry, qty))
                    continue
                consume = min(qty, need_tools)
                qty_left = qty - consume
                need_tools -= consume
                used_tools += consume
                if qty_left > 1e-9:
                    updated_pool.append((expiry, qty_left))
            tool_pool = updated_pool

        if need_trucks > 0 and truck_pool:
            updated_pool = []
            for expiry, qty in truck_pool:
                if need_trucks <= 0:
                    updated_pool.append((expiry, qty))
                    continue
                consume = min(qty, need_trucks)
                qty_left = qty - consume
                need_trucks -= consume
                used_trucks += consume
                if qty_left > 1e-9:
                    updated_pool.append((expiry, qty_left))
            truck_pool = updated_pool

        reused_tools[m] = used_tools
        reused_trucks[m] = used_trucks
        retained_trucks[m] = sum(q for _, q in truck_pool)

        tools_capex[m] = max(0.0, new_techs[m] - used_tools) * i.tools_per_new_tech_capex
        if i.capex_trucks_mode == "purchase_with_downpayment":
            truck_capex[m] = max(0.0, need_trucks) * i.truck_purchase_price * i.truck_downpayment_pct
        else:
            truck_capex[m] = 0.0

    trucks_for_cost = assigned_trucks + retained_trucks
    financed_trucks = trucks_for_cost * i.truck_financed_pct
    fleet_cost = trucks_for_cost * (
        i.fuel_per_truck_monthly + i.maint_per_truck_monthly + i.truck_insurance_per_truck_monthly
    ) * cost_mult + financed_trucks * i.truck_payment_monthly * cost_mult

    sales_payroll = sales_hours * i.sales_wage_per_hour * (1 + i.payroll_burden_pct) * cost_mult * sales_raise

    manager_active = _start_mask(dates, i.manager_start_year, i.manager_start_month)
    ops_active = _start_mask(dates, i.ops_manager_start_year, i.ops_manager_start_month)
    mkt_active = _start_mask(dates, i.marketing_manager_start_year, i.marketing_manager_start_month)
    manager_payroll = i.manager_salary_monthly * manager_active * cost_mult * mgr_raise
    ops_manager_payroll = i.ops_manager_salary_monthly * ops_active * cost_mult * ops_raise
    marketing_manager_payroll = i.marketing_manager_salary_monthly * mkt_active * cost_mult * mkt_raise
    management_payroll = manager_payroll + ops_manager_payroll + marketing_manager_payroll

    total_opex = fixed_opex + marketing_spend + fleet_cost + sales_payroll + management_payroll
    ebitda = gross_profit - total_opex

    ar_balance = total_revenue / 30 * i.ar_days
    ap_base = total_direct_costs + marketing_spend + fixed_opex + fleet_cost + sales_payroll + management_payroll
    ap_balance = ap_base / 30 * i.ap_days
    inventory_base = service_materials + repl_equipment + new_build_direct_cost
    inventory_balance = inventory_base / 30 * i.inventory_days
    nwc = ar_balance + inventory_balance - ap_balance
    opening_nwc = i.opening_ar_balance + i.opening_inventory_balance - i.opening_ap_balance
    change_nwc = np.diff(np.insert(nwc, 0, opening_nwc))
    operating_cash_flow = ebitda - change_nwc

    gross_capex = tools_capex + truck_capex
    capex = gross_capex - salvage_proceeds
    free_cash_flow = operating_cash_flow - capex

    loan_payment = np.zeros(horizon)
    loan_interest = np.zeros(horizon)
    loan_principal_paid = np.zeros(horizon)
    loan_balance = np.zeros(horizon)
    if i.enable_term_loan:
        payment = _term_payment(i.loan_principal, i.loan_annual_rate, int(i.loan_term_months))
        bal = i.loan_principal
        r = i.loan_annual_rate / 12
        for m in range(horizon):
            if bal <= 1e-6:
                break
            interest = bal * r
            principal = min(payment - interest, bal)
            debt_pmt = interest + principal
            bal = max(0.0, bal - principal)
            loan_payment[m] = debt_pmt
            loan_interest[m] = interest
            loan_principal_paid[m] = principal
            loan_balance[m] = bal

    loc_draw = np.zeros(horizon)
    loc_repay = np.zeros(horizon)
    loc_interest = np.zeros(horizon)
    loc_balance = np.zeros(horizon)
    owner_distributions = np.zeros(horizon)
    begin_cash = np.zeros(horizon)
    end_cash = np.zeros(horizon)
    net_cash_flow = np.zeros(horizon)

    cash = i.starting_cash
    loc_bal = 0.0
    loc_r = i.loc_annual_rate / 12
    for m in range(horizon):
        begin_cash[m] = cash
        loc_int = loc_bal * loc_r if i.enable_loc else 0.0
        loc_interest[m] = loc_int

        financing_cf_pre_loc = -loan_payment[m] - loc_int
        pre_loc_cash = cash + free_cash_flow[m] + financing_cf_pre_loc

        if i.enable_loc and pre_loc_cash < i.min_cash_target and loc_bal < i.loc_limit:
            draw = min(i.loc_limit - loc_bal, i.min_cash_target - pre_loc_cash)
            loc_draw[m] = max(0.0, draw)
            loc_bal += loc_draw[m]
            pre_loc_cash += loc_draw[m]

        if i.enable_loc and pre_loc_cash > (i.min_cash_target + i.loc_repay_buffer) and loc_bal > 0:
            repay = min(loc_bal, pre_loc_cash - (i.min_cash_target + i.loc_repay_buffer))
            loc_repay[m] = max(0.0, repay)
            loc_bal -= loc_repay[m]
            pre_loc_cash -= loc_repay[m]

        if i.enable_distributions and pre_loc_cash > i.distributions_only_if_cash_above:
            owner_distributions[m] = i.distributions_pct_of_ebitda * max(0.0, ebitda[m])
            protected_cash_floor = max(i.distributions_only_if_cash_above, i.min_cash_target)
            owner_distributions[m] = min(owner_distributions[m], max(0.0, pre_loc_cash - protected_cash_floor))

        cash = pre_loc_cash - owner_distributions[m]
        loc_balance[m] = loc_bal
        net_financing_cf = -loan_payment[m] - loc_interest[m] + loc_draw[m] - loc_repay[m] - owner_distributions[m]
        net_cash_flow[m] = operating_cash_flow[m] - capex[m] + net_financing_cf
        end_cash[m] = cash

    net_disbursements = capex + loan_payment + loc_interest + loc_repay + owner_distributions

    df = pd.DataFrame(
        {
            "Year": ((t // 12) + 1).astype(int),
            "Month_Number": (t + 1).astype(int),
            "Date": dates,
            "Year_Month_Label": dates.strftime("%Y-%m"),
            "Workdays": workdays,
            "Techs": techs,
            "Sales Staff": sales_staff,
            "Tech Hours": tech_hours,
            "Sales Hours": sales_hours,
            "New Tech Hires": new_techs,
            "Tech Attrition": tech_attrition,
            "New Sales Hires": new_sales,
            "Sales Attrition": sales_attrition,
            "Trucks": assigned_trucks,
            "Retained Trucks": retained_trucks,
            "Calls": calls,
            "Res Calls": res_calls,
            "LC Calls": lc_calls,
            "Service Revenue": service_rev,
            "Replacement Leads": repl_leads,
            "Replacement Jobs": repl_jobs,
            "Replacement Revenue": repl_rev,
            "Res Maintenance Agreements": res_maint_agreements,
            "LC Maintenance Agreements": lc_maint_agreements,
            "Maintenance Agreements": maint_agreements,
            "Res New Agreements": res_new_agreements,
            "LC New Agreements": lc_new_agreements,
            "Res Maintenance Visits": res_maint_visits,
            "LC Maintenance Visits": lc_maint_visits,
            "Res Maintenance Revenue": res_maint_rev,
            "LC Maintenance Revenue": lc_maint_rev,
            "Maintenance Revenue": maint_rev,
            "Res Service Upsell Revenue": res_service_upsell_rev,
            "LC Service Upsell Revenue": lc_service_upsell_rev,
            "Res Maintenance Upsell Revenue": res_maint_upsell_rev,
            "LC Maintenance Upsell Revenue": lc_maint_upsell_rev,
            "Upsell Revenue": upsell_revenue,
            "Res New Build Installs": res_new_build_installs,
            "LC New Build Installs": lc_new_build_installs,
            "Res New Build Revenue": res_new_build_revenue,
            "LC New Build Revenue": lc_new_build_revenue,
            "New Build Revenue": new_build_revenue,
            "Total Revenue": total_revenue,
            "Financing Fee Cost": financing_fee_cost,
            "Service Materials": service_materials,
            "Replacement Equipment": repl_equipment,
            "Permits": permits,
            "Disposal": disposal,
            "Tech Labor Cost per Wage Unit": tech_labor_cost_per_wage_unit,
            "Direct Labor": direct_labor,
            "Res Maintenance Direct Cost": res_maint_direct_cost,
            "LC Maintenance Direct Cost": lc_maint_direct_cost,
            "Maintenance Direct Cost": maint_direct_cost,
            "Upsell Direct Cost": upsell_direct_cost,
            "Res New Build Direct Cost": res_new_build_direct_cost,
            "LC New Build Direct Cost": lc_new_build_direct_cost,
            "New Build Direct Cost": new_build_direct_cost,
            "Total Direct Costs": total_direct_costs,
            "Gross Profit": gross_profit,
            "Fixed OPEX": fixed_opex,
            "Marketing Spend": marketing_spend,
            "Fleet Cost": fleet_cost,
            "Sales Payroll": sales_payroll,
            "Manager Payroll": manager_payroll,
            "Ops Manager Payroll": ops_manager_payroll,
            "Marketing Manager Payroll": marketing_manager_payroll,
            "Management Payroll": management_payroll,
            "Total OPEX": total_opex,
            "EBITDA": ebitda,
            "AR Balance": ar_balance,
            "AP Balance": ap_balance,
            "Inventory Balance": inventory_balance,
            "NWC": nwc,
            "Change in NWC": change_nwc,
            "Operating Cash Flow": operating_cash_flow,
            "Tools Capex": tools_capex,
            "Truck Capex": truck_capex,
            "Asset Salvage Proceeds": salvage_proceeds,
            "Gross Capex": gross_capex,
            "Capex": capex,
            "Free Cash Flow": free_cash_flow,
            "Term Loan Payment": loan_payment,
            "Term Loan Interest": loan_interest,
            "Term Loan Principal": loan_principal_paid,
            "Term Loan Balance": loan_balance,
            "LOC Draw": loc_draw,
            "LOC Repay": loc_repay,
            "LOC Interest": loc_interest,
            "LOC Balance": loc_balance,
            "Owner Distributions": owner_distributions,
            "Net Financing Cash Flow": -loan_payment - loc_interest + loc_draw - loc_repay - owner_distributions,
            "Net Cash Flow": net_cash_flow,
            "Begin Cash": begin_cash,
            "End Cash": end_cash,
            "Total Disbursements": net_disbursements,
            "Reused Tool Sets": reused_tools,
            "Reused Truck Units": reused_trucks,
        }
    )

    return df
