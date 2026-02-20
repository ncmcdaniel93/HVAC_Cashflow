"""Core HVAC cash flow model engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import floor
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class ModelInputs:
    data: Dict

    def __getattr__(self, key):
        if key in self.data:
            return self.data[key]
        raise AttributeError(key)


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
        "churn_annual_pct",
        "truck_downpayment_pct",
        "truck_financed_pct",
        "loan_annual_rate",
        "loc_annual_rate",
        "distributions_pct_of_ebitda",
    ]
    for field in pct_fields:
        val = inputs[field]
        if not (0 <= val <= 1):
            raise ValueError(f"{field} must be in [0,1].")

    non_negative_fields = [
        "ar_days",
        "ap_days",
        "inventory_days",
        "starting_techs",
        "tech_hire_per_quarter",
        "max_techs",
        "horizon_months",
        "start_month",
        "start_year",
    ]
    for field in non_negative_fields:
        if inputs[field] < 0:
            raise ValueError(f"{field} must be non-negative.")



def _term_payment(principal: float, annual_rate: float, term_months: int) -> float:
    if principal <= 0 or term_months <= 0:
        return 0.0
    r = annual_rate / 12
    if r == 0:
        return principal / term_months
    return principal * r / (1 - (1 + r) ** (-term_months))


def run_model(raw_inputs: Dict) -> pd.DataFrame:
    validate_inputs(raw_inputs)
    i = ModelInputs(raw_inputs)

    start_date = datetime(int(i.start_year), int(i.start_month), 1)
    dates = pd.date_range(start=start_date, periods=int(i.horizon_months), freq="MS")
    t = np.arange(i.horizon_months)

    techs = np.minimum(i.max_techs, i.starting_techs + (t // 3) * i.tech_hire_per_quarter)
    months = dates.month.values
    seasonality = 1 + i.seasonality_amplitude * np.sin(2 * np.pi * (months - i.peak_month) / 12)
    price_mult = (1 + i.monthly_price_growth) ** t
    cost_mult = (1 + i.monthly_cost_inflation) ** t

    calls = techs * i.calls_per_tech_per_day * i.work_days_per_month
    service_rev = calls * i.avg_service_ticket * seasonality * price_mult

    repl_leads = techs * i.repl_leads_per_tech_per_month
    repl_jobs = repl_leads * i.repl_close_rate
    repl_rev = repl_jobs * i.avg_repl_ticket * seasonality * price_mult

    if i.enable_maintenance:
        agreements = np.maximum(
            0,
            i.agreements_start * ((1 - i.churn_annual_pct) ** (t / 12))
            + i.new_agreements_per_month * (t + 1),
        )
        maint_rev = agreements * i.maint_monthly_fee * price_mult
        maint_direct_cost = agreements * (i.maint_visits_per_agreement_per_year / 12) * i.cost_per_maint_visit * cost_mult
    else:
        agreements = np.zeros(i.horizon_months)
        maint_rev = np.zeros(i.horizon_months)
        maint_direct_cost = np.zeros(i.horizon_months)

    total_revenue = service_rev + repl_rev + maint_rev
    financing_fee_cost = repl_rev * i.financing_penetration * i.financing_fee_pct

    service_materials = service_rev * i.service_material_pct * cost_mult
    repl_equipment = repl_rev * i.repl_equipment_pct * cost_mult
    permits = repl_jobs * i.permit_cost_per_repl_job * cost_mult
    disposal = repl_jobs * i.disposal_cost_per_repl_job * cost_mult
    direct_labor = techs * i.avg_hours_per_tech_per_month * i.tech_wage_per_hour * (1 + i.payroll_burden_pct) * cost_mult

    total_direct_costs = (
        service_materials + repl_equipment + permits + disposal + direct_labor + maint_direct_cost + financing_fee_cost
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
        np.full(i.horizon_months, i.paid_leads_per_month)
        if i.paid_leads_mode == "fixed"
        else techs * i.paid_leads_per_tech_per_month
    )
    marketing_spend = paid_leads * i.cost_per_lead * cost_mult + i.branding_fixed_monthly * cost_mult

    trucks = techs * i.trucks_per_tech
    fleet_cost = trucks * (
        i.truck_payment_monthly + i.fuel_per_truck_monthly + i.maint_per_truck_monthly + i.truck_insurance_per_truck_monthly
    ) * cost_mult

    total_opex = fixed_opex + marketing_spend + fleet_cost
    ebitda = gross_profit - total_opex

    ar_balance = total_revenue / 30 * i.ar_days
    direct_minus_labor = service_materials + repl_equipment + permits + disposal + maint_direct_cost
    ap_balance = direct_minus_labor / 30 * i.ap_days
    inventory_balance = repl_equipment / 30 * i.inventory_days
    nwc = ar_balance + inventory_balance - ap_balance
    change_nwc = np.diff(np.insert(nwc, 0, 0.0))
    operating_cash_flow = ebitda - change_nwc

    tech_change = np.diff(np.insert(techs, 0, techs[0]))
    new_techs = np.maximum(tech_change, 0)
    tools_capex = new_techs * i.tools_per_new_tech_capex
    truck_change = np.diff(np.insert(trucks, 0, trucks[0]))
    new_trucks = np.maximum(truck_change, 0)
    capex_trucks = (
        new_trucks * i.truck_purchase_price * i.truck_downpayment_pct
        if i.capex_trucks_mode == "purchase_with_downpayment"
        else np.zeros(i.horizon_months)
    )
    capex = tools_capex + capex_trucks
    free_cash_flow = operating_cash_flow - capex

    loan_payment = np.zeros(i.horizon_months)
    loan_interest = np.zeros(i.horizon_months)
    loan_principal_paid = np.zeros(i.horizon_months)
    loan_balance = np.zeros(i.horizon_months)

    if i.enable_term_loan:
        payment = _term_payment(i.loan_principal, i.loan_annual_rate, int(i.loan_term_months))
        bal = i.loan_principal
        r = i.loan_annual_rate / 12
        for m in range(i.horizon_months):
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

    loc_draw = np.zeros(i.horizon_months)
    loc_repay = np.zeros(i.horizon_months)
    loc_interest = np.zeros(i.horizon_months)
    loc_balance = np.zeros(i.horizon_months)
    owner_distributions = np.zeros(i.horizon_months)
    begin_cash = np.zeros(i.horizon_months)
    end_cash = np.zeros(i.horizon_months)
    net_cash_flow = np.zeros(i.horizon_months)

    cash = i.starting_cash
    loc_bal = 0.0
    loc_r = i.loc_annual_rate / 12

    for m in range(i.horizon_months):
        begin_cash[m] = cash
        loc_int = loc_bal * loc_r if i.enable_loc else 0.0
        loc_interest[m] = loc_int

        financing_cf = -loan_payment[m] - loc_int
        pre_loc_cash = cash + free_cash_flow[m] + financing_cf

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
            owner_distributions[m] = min(owner_distributions[m], max(0.0, pre_loc_cash))

        cash = pre_loc_cash - owner_distributions[m]

        loc_balance[m] = loc_bal
        net_financing_cf = -loan_payment[m] - loc_interest[m] + loc_draw[m] - loc_repay[m] - owner_distributions[m]
        net_cash_flow[m] = operating_cash_flow[m] - capex[m] + net_financing_cf
        end_cash[m] = cash

    df = pd.DataFrame(
        {
            "Year": ((t // 12) + 1).astype(int),
            "Month_Number": (t + 1).astype(int),
            "Date": dates,
            "Year_Month_Label": dates.strftime("%Y-%m"),
            "Techs": techs,
            "Calls": calls,
            "Service Revenue": service_rev,
            "Replacement Leads": repl_leads,
            "Replacement Jobs": repl_jobs,
            "Replacement Revenue": repl_rev,
            "Maintenance Agreements": agreements,
            "Maintenance Revenue": maint_rev,
            "Total Revenue": total_revenue,
            "Financing Fee Cost": financing_fee_cost,
            "Service Materials": service_materials,
            "Replacement Equipment": repl_equipment,
            "Permits": permits,
            "Disposal": disposal,
            "Direct Labor": direct_labor,
            "Maintenance Direct Cost": maint_direct_cost,
            "Total Direct Costs": total_direct_costs,
            "Gross Profit": gross_profit,
            "Fixed OPEX": fixed_opex,
            "Marketing Spend": marketing_spend,
            "Fleet Cost": fleet_cost,
            "Total OPEX": total_opex,
            "EBITDA": ebitda,
            "AR Balance": ar_balance,
            "AP Balance": ap_balance,
            "Inventory Balance": inventory_balance,
            "NWC": nwc,
            "Change in NWC": change_nwc,
            "Operating Cash Flow": operating_cash_flow,
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
            "Net Cash Flow": net_cash_flow,
            "Begin Cash": begin_cash,
            "End Cash": end_cash,
        }
    )

    return df
