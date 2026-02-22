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


INPUT_CALCULATION_DETAILS: dict[str, str] = {
    "calls_per_tech_per_day": "Used in monthly demand: Calls = Techs x Workdays x Calls/Tech/Day x seasonality.",
    "res_capacity_pct": "Splits total calls into residential vs light commercial demand before segment revenue math.",
    "avg_service_ticket": "Service Revenue = Calls x Average Service Ticket x monthly price factor.",
    "service_material_pct": "Service Materials = Service Revenue x service material %. Affects gross profit.",
    "attach_rate": "New-customer proxy in CAC uses Calls x attach rate plus replacement jobs.",
    "repl_leads_per_tech_per_month": "Replacement Leads = Techs x leads/tech/month x seasonality.",
    "repl_close_rate": "Replacement Jobs = Replacement Leads x effective close rate (plus sales lift, capped).",
    "sales_repl_close_lift_per_fte": "Raises replacement close rate as sales headcount grows; limited by close-rate cap.",
    "sales_repl_close_rate_cap": "Upper bound on effective replacement close rate after sales-lift adjustments.",
    "avg_repl_ticket": "Replacement Revenue = Replacement Jobs x average replacement ticket x price factor.",
    "repl_equipment_pct": "Replacement Equipment cost = Replacement Revenue x equipment %. Affects margin.",
    "permit_cost_per_repl_job": "Permits = Replacement Jobs x permit cost per replacement job.",
    "disposal_cost_per_repl_job": "Disposal = Replacement Jobs x disposal cost per replacement job.",
    "financing_penetration": "Financing Fee Cost = Replacement Revenue x financing penetration x financing fee %.",
    "financing_fee_pct": "Financing Fee Cost = Replacement Revenue x financing penetration x financing fee %.",
    "starting_techs": "Seeds monthly technician capacity, labor hours, and call throughput.",
    "max_techs": "Caps staffing growth in the monthly tech headcount trajectory.",
    "tech_staffing_events": "Monthly hires/attrition events shift tech headcount and capacity over time.",
    "sales_starting_staff": "Seeds monthly sales staffing used in close-rate lift and sales payroll.",
    "sales_staffing_events": "Monthly hires/attrition events shift sales headcount and payroll trajectory.",
    "tech_hours_per_day": "Tech Hours = Techs x Workdays x tech hours/day; feeds direct labor and break-even rates.",
    "sales_hours_per_day": "Sales Hours = Sales Staff x Workdays x sales hours/day; feeds sales payroll.",
    "tech_wage_per_hour": "Direct Labor = Tech Hours x tech wage x (1 + payroll burden).",
    "sales_wage_per_hour": "Sales Payroll = Sales Hours x sales wage x (1 + payroll burden).",
    "payroll_burden_pct": "Payroll burden multiplies tech and sales wage cost lines.",
    "tools_per_new_tech_capex": "Tools Capex = net new tech hires requiring non-reused tools x tools-per-tech capex.",
    "asset_reuse_lag_months": "Controls how long tools/trucks from attrition remain reusable before expiry/salvage.",
    "asset_expiry_mode": "Changes post-expiry handling of pooled assets: release, retain, or salvage.",
    "asset_salvage_pct": "Asset Salvage Proceeds = expired pooled asset value x salvage %. Reduces net capex.",
    "enable_maintenance": "Turns maintenance agreement revenue and direct-cost logic on/off.",
    "maint_visits_capacity_per_tech_per_month": "Constrains maintenance visits by technician capacity each month.",
    "res_agreements_start": "Starting residential agreements feed recurring maintenance revenue at month 1.",
    "res_new_agreements_per_month": "Adds monthly residential agreement growth before churn adjustments.",
    "res_churn_annual_pct": "Converts to monthly churn; reduces residential agreement retention each month.",
    "res_maint_monthly_fee": "Residential maintenance recurring revenue uses monthly fee x active agreements.",
    "lc_agreements_start": "Starting light-commercial agreements feed recurring maintenance revenue at month 1.",
    "lc_new_agreements_per_month": "Adds monthly light-commercial agreement growth before churn adjustments.",
    "lc_churn_annual_pct": "Converts to monthly churn; reduces LC agreement retention each month.",
    "lc_maint_quarterly_fee": "LC maintenance recurring revenue uses quarterly fee cadence on active agreements.",
    "res_service_upsell_conversion_pct": "Upsell jobs from residential service visits = visits x conversion %. ",
    "res_service_upsell_revenue_per_visit": "Residential service upsell revenue = converted visits x revenue per conversion.",
    "res_service_upsell_gross_margin_pct": "Residential service upsell direct cost back-solves from gross margin %. ",
    "lc_service_upsell_conversion_pct": "LC service upsell jobs = LC service visits x conversion %. ",
    "lc_service_upsell_revenue_per_visit": "LC service upsell revenue = converted visits x revenue per conversion.",
    "lc_service_upsell_gross_margin_pct": "LC service upsell direct cost back-solves from gross margin %. ",
    "res_maint_upsell_conversion_pct": "Residential maintenance upsell jobs = maintenance visits x conversion %. ",
    "res_maint_upsell_revenue_per_visit": "Residential maintenance upsell revenue = conversions x revenue per conversion.",
    "res_maint_upsell_gross_margin_pct": "Residential maintenance upsell direct cost back-solves from gross margin %. ",
    "lc_maint_upsell_conversion_pct": "LC maintenance upsell jobs = LC maintenance visits x conversion %. ",
    "lc_maint_upsell_revenue_per_visit": "LC maintenance upsell revenue = conversions x revenue per conversion.",
    "lc_maint_upsell_gross_margin_pct": "LC maintenance upsell direct cost back-solves from gross margin %. ",
    "new_build_mode": "Selects install-volume logic path: schedule, seasonal monthly baseline, or annual-total allocation.",
    "res_new_build_avg_ticket": "Residential new-build revenue = installs x avg ticket x price factor.",
    "lc_new_build_avg_ticket": "LC new-build revenue = installs x avg ticket x price factor.",
    "res_new_build_gross_margin_pct": "Residential new-build direct cost back-solves from gross margin %. ",
    "lc_new_build_gross_margin_pct": "LC new-build direct cost back-solves from gross margin %. ",
    "res_new_build_install_schedule": "When new-build mode is schedule, this monthly series directly sets residential installs.",
    "lc_new_build_install_schedule": "When new-build mode is schedule, this monthly series directly sets LC installs.",
    "paid_leads_mode": "Marketing lead volume uses fixed monthly leads or scales by tech count.",
    "paid_leads_per_month": "Used when paid-leads mode is fixed: marketing spend = leads x cost/lead + branding fixed.",
    "paid_leads_per_tech_per_month": "Used when lead mode is per-tech: leads = techs x leads/tech/month.",
    "cost_per_lead": "Marketing spend = paid leads x cost per lead + branding fixed.",
    "branding_fixed_monthly": "Adds fixed monthly brand spend on top of variable lead-acquisition spend.",
    "trucks_per_tech": "Assigned Trucks = Techs x trucks/tech; drives fleet opex and truck-financing exposure.",
    "truck_payment_monthly": "Part of fleet cost for financed portion of trucks (truck payment x financed trucks).",
    "fuel_per_truck_monthly": "Part of fleet opex: trucks-for-cost x fuel/truck/month.",
    "maint_per_truck_monthly": "Part of fleet opex: trucks-for-cost x maintenance/truck/month.",
    "truck_insurance_per_truck_monthly": "Part of fleet opex: trucks-for-cost x truck insurance/truck/month.",
    "truck_purchase_price": "Truck capex basis for downpayment mode and salvage-value basis on expiry.",
    "capex_trucks_mode": "Controls truck capex path: payment-only vs purchase with downpayment.",
    "truck_downpayment_pct": "Truck Capex = required new trucks x purchase price x downpayment %. ",
    "truck_financed_pct": "Financed truck share scales truck-payment component of fleet opex.",
    "office_payroll_monthly": "Feeds fixed OPEX and therefore EBITDA, operating cash flow, and cash runway.",
    "rent_monthly": "Feeds fixed OPEX and therefore EBITDA, operating cash flow, and cash runway.",
    "utilities_monthly": "Feeds fixed OPEX and therefore EBITDA, operating cash flow, and cash runway.",
    "insurance_monthly": "Feeds fixed OPEX and therefore EBITDA, operating cash flow, and cash runway.",
    "software_monthly": "Feeds fixed OPEX and therefore EBITDA, operating cash flow, and cash runway.",
    "other_fixed_monthly": "Feeds fixed OPEX and therefore EBITDA, operating cash flow, and cash runway.",
    "manager_salary_monthly": "Manager payroll line starts at manager start date and flows into management payroll.",
    "ops_manager_salary_monthly": "Ops-manager payroll starts at configured start date and flows into management payroll.",
    "marketing_manager_salary_monthly": "Marketing-manager payroll starts at configured start date and flows into management payroll.",
    "annual_raise_pct_tech": "Raises tech wage trajectory yearly from effective month; increases direct labor over time.",
    "annual_raise_pct_sales": "Raises sales wage trajectory yearly from effective month; increases sales payroll.",
    "ar_days": "AR Balance = revenue / 30 x AR days; influences NWC and operating cash timing.",
    "ap_days": "AP Balance = payable base / 30 x AP days; offsets working-capital cash drag.",
    "inventory_days": "Inventory Balance = inventory base / 30 x inventory days; increases NWC when higher.",
    "starting_cash": "Initial month begin-cash baseline for full liquidity runway simulation.",
    "enable_term_loan": "Turns term-loan amortization schedule on/off (interest + principal cash outflows).",
    "loan_principal": "Initial debt balance for term-loan amortization and debt-service cash flow.",
    "loan_annual_rate": "Term-loan interest component each month uses annual rate / 12.",
    "loan_term_months": "Defines amortization horizon and monthly principal pace for term debt.",
    "enable_loc": "Turns line-of-credit draw/repay logic and LOC interest on/off.",
    "loc_limit": "Maximum LOC balance available for liquidity support.",
    "loc_annual_rate": "LOC interest = LOC balance x (annual rate / 12).",
    "min_cash_target": "If projected cash falls below target, model attempts LOC draw up to limit.",
    "loc_repay_buffer": "Repayment trigger buffer above min-cash target before LOC paydown.",
    "enable_distributions": "Turns owner distribution cash outflow logic on/off.",
    "distributions_pct_of_ebitda": "Owner distribution target = EBITDA x distribution %. Subject to cash floor guardrails.",
    "distributions_only_if_cash_above": "Distribution paid only when cash exceeds this threshold (and min-cash protection).",
    "monthly_price_growth": "Applies monthly escalation factor to revenue-side price/ticket assumptions.",
    "monthly_cost_inflation": "Applies monthly escalation factor to cost-side assumptions.",
    "discount_rate_annual_nominal": "Used only for present-value display mode; does not change nominal operations.",
    "value_mode": "Display transform: nominal, real inflation-adjusted, or real present value.",
}


INPUT_IMPACT_DETAILS: dict[str, str] = {
    "calls_per_tech_per_day": "Higher values usually raise service revenue and lead flow, but can stress capacity realism.",
    "avg_service_ticket": "Higher tickets increase revenue directly and usually improve gross profit if cost % is stable.",
    "service_material_pct": "Higher material % compresses gross margin and EBITDA.",
    "repl_close_rate": "Higher close rates shift more leads into replacement jobs and revenue.",
    "avg_repl_ticket": "Higher replacement ticket boosts revenue and contribution margin.",
    "tech_wage_per_hour": "Higher tech wage reduces EBITDA and free cash flow unless price/volume offsets exist.",
    "sales_wage_per_hour": "Higher sales wage raises OPEX and lowers EBITDA if close-rate lift does not compensate.",
    "payroll_burden_pct": "Higher burden increases both direct labor and sales payroll.",
    "cost_per_lead": "Higher lead cost increases marketing spend and CAC; reduces EBITDA/cash.",
    "truck_payment_monthly": "Higher payment increases fleet cost and total disbursement pressure.",
    "fuel_per_truck_monthly": "Higher fuel increases fleet opex and reduces cash generation.",
    "maint_per_truck_monthly": "Higher truck maintenance increases fleet opex and reduces cash generation.",
    "truck_insurance_per_truck_monthly": "Higher truck insurance increases fleet opex and reduces cash generation.",
    "ar_days": "Higher AR days slows collections and typically reduces operating cash flow.",
    "ap_days": "Higher AP days delays payments and usually improves near-term cash (with supplier-trust tradeoff).",
    "inventory_days": "Higher inventory days ties up working capital and lowers near-term cash.",
    "loan_annual_rate": "Higher debt rates increase interest outflow and reduce net cash flow.",
    "loc_annual_rate": "Higher LOC rate increases financing cost when LOC is utilized.",
    "min_cash_target": "Higher target improves buffer safety but can increase LOC draw frequency.",
    "distributions_pct_of_ebitda": "Higher payout % increases owner cash outflow and can reduce retained liquidity.",
    "distributions_only_if_cash_above": "Higher threshold is more conservative and protects liquidity.",
    "monthly_price_growth": "Higher price growth raises future revenue trajectories if demand assumptions hold.",
    "monthly_cost_inflation": "Higher cost inflation raises direct/OPEX lines and can pressure margins.",
}


def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _normalize_key(key: str) -> str:
    return str(key or "").strip()


def _pattern_calculation_detail(key: str) -> str:
    k = _normalize_key(key).lower()
    if not k:
        return ""
    if k.endswith("_wage_per_hour"):
        return "Used as hourly wage input in payroll formulas (hours x wage x (1 + burden))."
    if k.endswith("_salary_monthly"):
        return "Used as monthly management payroll line item, subject to start-date activation."
    if "churn" in k:
        return "Converted into monthly retention decay and applied to agreement balance roll-forward."
    if "conversion" in k or "close_rate" in k:
        return "Acts as a conversion coefficient from opportunities/visits into closed jobs or upsell events."
    if k.endswith("_gross_margin_pct"):
        return "Used to back-solve direct cost from revenue via direct cost = revenue x (1 - gross margin %)."
    if k.endswith("_avg_ticket") or "ticket" in k:
        return "Applied as revenue per unit (job/visit/install) in segment revenue calculations."
    if "cost_per" in k or k.endswith("_monthly") and any(t in k for t in ("rent", "utilities", "insurance", "software", "other_fixed")):
        return "Feeds monthly cost lines directly, then rolls into OPEX, EBITDA, and cash flow."
    if k.endswith("_days"):
        return "Used in balance-sheet timing (value / 30 x days) to compute AR/AP/Inventory working-capital balances."
    if "raise_pct" in k:
        return "Applied annually from effective month to wage/salary trajectory multipliers."
    if "mode" in k:
        return "Selects one of multiple calculation branches in the model logic."
    if "rate" in k or k.endswith("_pct"):
        return "Used as a percentage/rate coefficient within the related calculation path."
    return ""


def _pattern_impact_detail(key: str) -> str:
    k = _normalize_key(key).lower()
    if not k:
        return ""
    if any(t in k for t in ("wage", "salary", "cost", "payment", "insurance", "fuel", "rent", "utilities", "software")):
        return "Higher values generally increase costs and reduce EBITDA/free cash flow."
    if any(t in k for t in ("ticket", "revenue", "calls", "close", "conversion", "agreement", "installs", "lead")):
        return "Higher values generally increase growth/revenue potential, with related cost and capacity effects."
    if "days" in k:
        return "Changes cash timing through working-capital balances rather than pure P&L revenue/cost levels."
    if "mode" in k:
        return "Switching modes can materially change output behavior and interpretation."
    return ""


def calculation_logic_detail(key: str) -> str:
    k = _normalize_key(key)
    if not k:
        return ""
    return INPUT_CALCULATION_DETAILS.get(k, _pattern_calculation_detail(k))


def impact_detail(key: str) -> str:
    k = _normalize_key(key)
    if not k:
        return ""
    return INPUT_IMPACT_DETAILS.get(k, _pattern_impact_detail(k))


def help_with_guidance(key: str, base_help: str) -> str:
    g = INPUT_GUIDANCE.get(key)
    calc = calculation_logic_detail(key)
    impact = impact_detail(key)
    parts = [str(base_help or "").strip()]
    if g:
        parts.append(f"Reasonable range: {_fmt(g['min'])} to {_fmt(g['max'])}.")
        if str(g.get("note", "")).strip():
            parts.append(str(g["note"]).strip())
    if calc:
        parts.append(f"Calculation use: {calc}")
    if impact:
        parts.append(f"Impact: {impact}")
    merged = " ".join(p for p in parts if p).strip()
    merged = " ".join(merged.split())
    if len(merged) > 900:
        return f"{merged[:897].rstrip()}..."
    return merged


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
