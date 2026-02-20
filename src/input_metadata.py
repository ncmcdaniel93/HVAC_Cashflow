"""Input guidance metadata and advisory range checks."""

from __future__ import annotations

from typing import Any


INPUT_GUIDANCE: dict[str, dict[str, Any]] = {
    "horizon_months": {"min": 12, "max": 120, "note": "Typical planning horizon is 24 to 60 months."},
    "seasonality_amplitude": {"min": 0.0, "max": 0.3, "note": "Most HVAC businesses operate within modest seasonal amplitude."},
    "calls_per_tech_per_day": {"min": 1.0, "max": 5.0, "note": "Typical completed calls per tech workday."},
    "tech_hours_per_day": {"min": 6.0, "max": 10.0, "note": "Field tech paid hours per workday."},
    "sales_hours_per_day": {"min": 6.0, "max": 10.0, "note": "Sales staff paid hours per workday."},
    "payroll_burden_pct": {"min": 0.1, "max": 0.35, "note": "Payroll taxes/benefits load often falls in this range."},
    "tech_wage_per_hour": {"min": 20.0, "max": 70.0, "note": "Loaded HVAC technician base hourly wage before burden."},
    "sales_wage_per_hour": {"min": 18.0, "max": 65.0, "note": "Hourly wage for sales staff."},
    "tools_per_new_tech_capex": {"min": 1000.0, "max": 8000.0, "note": "Upfront tool investment per newly hired technician."},
    "asset_reuse_lag_months": {"min": 0, "max": 12, "note": "Retention period before unused assets expire or are salvaged."},
    "avg_service_ticket": {"min": 150.0, "max": 900.0, "note": "Typical residential service invoice ticket range."},
    "service_material_pct": {"min": 0.05, "max": 0.35, "note": "Service material spend as a share of service revenue."},
    "repl_leads_per_tech_per_month": {"min": 1.0, "max": 8.0, "note": "Monthly replacement opportunities generated per tech."},
    "avg_repl_ticket": {"min": 5000.0, "max": 25000.0, "note": "Typical replacement ticket for residential/light commercial mix."},
    "repl_equipment_pct": {"min": 0.3, "max": 0.7, "note": "Replacement equipment/materials as a revenue share."},
    "cost_per_lead": {"min": 20.0, "max": 300.0, "note": "Paid lead costs vary by channel and geography."},
    "paid_leads_per_tech_per_month": {"min": 5.0, "max": 60.0, "note": "Paid lead volume scaling with technician count."},
    "branding_fixed_monthly": {"min": 0.0, "max": 30000.0, "note": "Fixed brand spend can be zero for lean scenarios."},
    "repl_close_rate": {"min": 0.2, "max": 0.8, "note": "Observed replacement close-rate envelope in many markets."},
    "sales_repl_close_lift_per_fte": {"min": 0.0, "max": 0.08, "note": "Incremental close-rate lift each sales FTE can add."},
    "sales_repl_close_rate_cap": {"min": 0.3, "max": 0.9, "note": "Upper bound on replacement close rate after lift."},
    "res_agreements_start": {"min": 0.0, "max": 3000.0, "note": "Starting residential maintenance agreement count."},
    "res_churn_annual_pct": {"min": 0.03, "max": 0.25, "note": "Annual residential maintenance churn."},
    "lc_churn_annual_pct": {"min": 0.02, "max": 0.2, "note": "Annual light commercial maintenance churn."},
    "res_maint_monthly_fee": {"min": 10.0, "max": 60.0, "note": "Monthly agreement fee for residential maintenance."},
    "lc_maint_quarterly_fee": {"min": 60.0, "max": 600.0, "note": "Quarterly agreement billing for light commercial."},
    "maint_visits_capacity_per_tech_per_month": {"min": 5.0, "max": 60.0, "note": "Maximum maintenance visits each tech can absorb monthly."},
    "res_service_upsell_conversion_pct": {"min": 0.02, "max": 0.25, "note": "Share of service visits converting to upsell."},
    "res_maint_upsell_conversion_pct": {"min": 0.03, "max": 0.35, "note": "Share of maintenance visits converting to upsell."},
    "res_service_upsell_revenue_per_visit": {"min": 100.0, "max": 1200.0, "note": "Average upsell revenue when residential service visit converts."},
    "lc_service_upsell_revenue_per_visit": {"min": 150.0, "max": 2500.0, "note": "Average upsell revenue when LC service visit converts."},
    "res_new_build_avg_ticket": {"min": 5000.0, "max": 30000.0, "note": "Typical residential new-build install ticket."},
    "lc_new_build_avg_ticket": {"min": 7000.0, "max": 50000.0, "note": "Typical light-commercial new-build install ticket."},
    "res_new_build_gross_margin_pct": {"min": 0.2, "max": 0.6, "note": "Gross margin percent on residential new-build installs."},
    "lc_new_build_gross_margin_pct": {"min": 0.15, "max": 0.55, "note": "Gross margin percent on LC new-build installs."},
    "trucks_per_tech": {"min": 0.7, "max": 1.2, "note": "Truck to technician deployment ratio."},
    "truck_payment_monthly": {"min": 200.0, "max": 2000.0, "note": "Monthly financed truck payment per active truck."},
    "truck_purchase_price": {"min": 25000.0, "max": 110000.0, "note": "Purchase cost for each replacement/new truck."},
    "office_payroll_monthly": {"min": 0.0, "max": 120000.0, "note": "Back-office payroll not in direct field labor."},
    "manager_salary_monthly": {"min": 0.0, "max": 25000.0, "note": "Monthly manager payroll cost when active."},
    "ops_manager_salary_monthly": {"min": 0.0, "max": 25000.0, "note": "Monthly operations manager payroll cost when active."},
    "marketing_manager_salary_monthly": {"min": 0.0, "max": 25000.0, "note": "Monthly marketing manager payroll cost when active."},
    "annual_raise_pct_tech": {"min": 0.0, "max": 0.1, "note": "Annual wage step-up for technicians."},
    "annual_raise_pct_sales": {"min": 0.0, "max": 0.1, "note": "Annual wage step-up for sales staff."},
    "monthly_price_growth": {"min": 0.0, "max": 0.02, "note": "Monthly pricing growth assumptions typically low single digits annualized."},
    "monthly_cost_inflation": {"min": 0.0, "max": 0.02, "note": "Monthly cost inflation assumptions typically low single digits annualized."},
    "discount_rate_annual_nominal": {"min": 0.03, "max": 0.2, "note": "Nominal annual discount rate for PV real-dollar view."},
    "loan_annual_rate": {"min": 0.03, "max": 0.18, "note": "Typical annualized borrowing rate for term debt."},
    "loc_annual_rate": {"min": 0.04, "max": 0.22, "note": "Typical annualized line-of-credit borrowing rate."},
    "min_cash_target": {"min": 0.0, "max": 500000.0, "note": "Operating cash floor the business aims to protect."},
    "ar_days": {"min": 0.0, "max": 90.0, "note": "Accounts receivable collection period."},
    "ap_days": {"min": 0.0, "max": 90.0, "note": "Accounts payable cycle length."},
    "inventory_days": {"min": 0.0, "max": 90.0, "note": "Inventory days on hand."},
}


def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def help_with_guidance(key: str, base_help: str) -> str:
    g = INPUT_GUIDANCE.get(key)
    if not g:
        return base_help
    return f"{base_help} Reasonable range: {_fmt(g['min'])} to {_fmt(g['max'])}. {g['note']}"


def advisory_warnings(inputs: dict) -> list[str]:
    warnings: list[str] = []
    for key, g in INPUT_GUIDANCE.items():
        if key not in inputs:
            continue
        try:
            v = float(inputs[key])
        except (TypeError, ValueError):
            continue
        if v < g["min"] or v > g["max"]:
            warnings.append(
                f"{key}={v:.3f} is outside the recommended range [{_fmt(g['min'])}, {_fmt(g['max'])}]."
            )
    return warnings
