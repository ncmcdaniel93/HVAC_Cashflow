```text
TITLE: Build a Streamlit HVAC Cash Flow Model UI (Single Scenario, 5-Year Forecast, Driver-Based, With Sensitivity)

ROLE
You are ChatGPT Codex acting as a senior full stack engineer and financial modeling engineer. Build a simple, reliable Streamlit app that lets a user input a single scenario of HVAC business drivers (scalar inputs) and generates:
1) a 5-year monthly forecast (60 months by default, adjustable),
2) a line-by-line cash flow calculation with intermediate subtotals and annotations,
3) charts,
4) key metrics,
5) one-way sensitivity analysis on the most important drivers.

NON-NEGOTIABLE REQUIREMENTS
- Keep it simple: one Streamlit app file is acceptable (app.py), plus an optional /src folder.
- No external services. No database. No authentication. Local execution only.
- Use Python only. Use pandas + numpy + plotly (or matplotlib) + streamlit.
- All time series must be generated from scalar inputs (no manual month-by-month input arrays).
- Model must be readable and auditable: show intermediate calculations and subtotals.
- No hardcoded magic numbers in the engine. All assumptions must come from UI defaults, with defaults defined in one config dict.
- Provide clear labels, units, and definitions for each input in the UI (tooltips or help text).
- Avoid long dashes in any user-visible text.
- Use consistent abbreviations and define them in UI (for example EBITDA defined as Earnings Before Interest, Taxes, Depreciation, and Amortization).
- Output must include a “Download CSV” for the monthly line-item table.

PROJECT OUTPUTS
Create these files:
- app.py (Streamlit UI + charts)
- src/model.py (core model functions)
- src/defaults.py (default assumptions dict)
- src/metrics.py (metric calculations)
- src/sensitivity.py (one-way sensitivity)
- requirements.txt
Optional:
- README.md (setup steps)

APP UX SPEC
Sidebar: Scenario Manager (all inputs)
Main: Tabs
  Tab 1: Summary Dashboard (headline KPIs + charts)
  Tab 2: Line-by-line Cash Flow (monthly table + grouped subtotals + annotations)
  Tab 3: Drivers (time series for key drivers: tech count, calls, leads, revenue split)
  Tab 4: Sensitivity (tornado chart + table)

HORIZON (UPDATED)
- Default forecast horizon is 60 months (5 years).
- User can adjust horizon_months from 12 to 120 via a sidebar control.
- All outputs (tables, charts, metrics, sensitivities) must adapt to horizon_months.

DATE HANDLING
- Start date is a user-selected month and year.
- Generate a monthly date index of length horizon_months.
- Include columns:
  Year (1..N),
  Month_Number (1..horizon_months),
  Year_Month_Label (YYYY-MM).

MODEL INPUTS (MUST IMPLEMENT IN SIDEBAR, WITH DEFINITIONS + DEFAULTS)
Model Controls
- start_year (int)
- start_month (1..12)
- horizon_months (12..120, default 60)
- monthly_price_growth (0.0% to 2.0%)
- monthly_cost_inflation (0.0% to 2.0%)
- seasonality_amplitude (0% to 40%)
- peak_month (1..12)

Staffing and Ops
- starting_techs
- tech_hire_per_quarter
- max_techs
- calls_per_tech_per_day
- work_days_per_month
- avg_hours_per_tech_per_month
- tech_wage_per_hour
- payroll_burden_pct
- tools_per_new_tech_capex

Service Revenue
- avg_service_ticket
- service_material_pct
- attach_rate (diagnostic to repair proxy, used for CAC proxy and optional service mix)

Replacement Revenue
- repl_leads_per_tech_per_month
- repl_close_rate
- avg_repl_ticket
- repl_equipment_pct
- permit_cost_per_repl_job
- disposal_cost_per_repl_job
- financing_penetration
- financing_fee_pct

Maintenance Program
- enable_maintenance (bool)
- agreements_start
- new_agreements_per_month
- churn_annual_pct
- maint_monthly_fee
- cost_per_maint_visit
- maint_visits_per_agreement_per_year

Marketing
- paid_leads_mode: fixed or per_tech
- paid_leads_per_month (if fixed)
- paid_leads_per_tech_per_month (if per_tech)
- cost_per_lead
- branding_fixed_monthly

Fleet
- trucks_per_tech
- truck_payment_monthly
- fuel_per_truck_monthly
- maint_per_truck_monthly
- truck_insurance_per_truck_monthly
- truck_purchase_price
- capex_trucks_mode: payments_only or purchase_with_downpayment
- truck_downpayment_pct
- truck_financed_pct

Overhead
- office_payroll_monthly
- rent_monthly
- utilities_monthly
- insurance_monthly
- software_monthly
- other_fixed_monthly

Working Capital
- ar_days
- ap_days
- inventory_days
- starting_cash

Debt
- enable_term_loan (bool)
- loan_principal
- loan_annual_rate
- loan_term_months
- enable_loc (bool)
- loc_limit
- loc_annual_rate
- min_cash_target
- loc_repay_buffer

Distributions
- enable_distributions (bool)
- distributions_pct_of_ebitda
- distributions_only_if_cash_above

MODEL LOGIC (MUST IMPLEMENT)
1) Time series generation from scalars
   - Technician count over time:
     techs[t] = min(max_techs, starting_techs + floor(t/3) * tech_hire_per_quarter) where t starts at 0
   - Seasonality multiplier (sinusoid):
     seasonality[t] = 1 + seasonality_amplitude * sin(2*pi*(month_index - peak_month)/12)
     where month_index is 1..12 from the calendar month
   - Growth multipliers:
     price_multiplier[t] = (1 + monthly_price_growth)^t
     cost_multiplier[t]  = (1 + monthly_cost_inflation)^t

2) Revenue model (monthly)
   - Service calls:
     calls = techs * calls_per_tech_per_day * work_days_per_month
     service_rev = calls * avg_service_ticket * seasonality * price_multiplier
   - Replacement:
     replacement_leads = techs * repl_leads_per_tech_per_month
     repl_jobs = replacement_leads * repl_close_rate
     repl_rev = repl_jobs * avg_repl_ticket * seasonality * price_multiplier
   - Maintenance (if enabled):
     agreements[t] = max(0, agreements_start*(1 - churn_annual_pct)^(t/12) + new_agreements_per_month * (t+1))
     maint_rev = agreements * maint_monthly_fee * price_multiplier
   - Total revenue = service_rev + repl_rev + maint_rev
   - Financing fees (merchant fee) reduce cash collected:
     financing_fee_cost = repl_rev * financing_penetration * financing_fee_pct

3) Direct costs (COGS)
   - Service materials:
     service_mat = service_rev * service_material_pct
   - Replacement equipment:
     repl_equip = repl_rev * repl_equipment_pct
   - Replacement permits and disposal:
     permits = repl_jobs * permit_cost_per_repl_job
     disposal = repl_jobs * disposal_cost_per_repl_job
   - Direct labor:
     fully_loaded_wage = tech_wage_per_hour * (1 + payroll_burden_pct)
     paid_hours = techs * avg_hours_per_tech_per_month
     direct_labor = paid_hours * fully_loaded_wage * cost_multiplier
   - Maintenance direct cost (if enabled):
     maint_visits_per_month = agreements * (maint_visits_per_agreement_per_year / 12)
     maint_cost = maint_visits_per_month * cost_per_maint_visit * cost_multiplier
   - Total direct costs = service_mat + repl_equip + permits + disposal + direct_labor + maint_cost + financing_fee_cost

4) Gross profit and EBITDA
   - Gross profit = revenue - total_direct_costs
   - Operating expenses (OPEX)
     fixed_opex = (office_payroll_monthly + rent_monthly + utilities_monthly + insurance_monthly + software_monthly + other_fixed_monthly) * cost_multiplier
     marketing:
       if paid_leads_mode == fixed:
         paid_leads = paid_leads_per_month
       else:
         paid_leads = techs * paid_leads_per_tech_per_month
       marketing_spend = paid_leads * cost_per_lead * cost_multiplier + branding_fixed_monthly * cost_multiplier
     fleet:
       trucks = techs * trucks_per_tech
       fleet_cost = trucks*(truck_payment_monthly + fuel_per_truck_monthly + maint_per_truck_monthly + truck_insurance_per_truck_monthly) * cost_multiplier
     total_opex = fixed_opex + marketing_spend + fleet_cost
   - EBITDA = gross_profit - total_opex

5) Working capital (simple balance method)
   - AR:
     ar_balance[t] = revenue[t] * (ar_days/30)
   - AP:
     ap_balance[t] = (total_direct_costs[t] + marketing_spend[t] + fixed_opex[t] + fleet_cost[t]) * (ap_days/30)
   - Inventory:
     inv_balance[t] = (service_mat[t] + repl_equip[t]) * (inventory_days/30)
   - Net working capital (NWC) = AR + Inventory - AP
   - Change in NWC = NWC[t] - NWC[t-1] (first month change uses starting balances as 0)
   - Cash impact from NWC = -Change in NWC

6) Debt and financing
   Term loan (if enabled):
   - monthly_rate = loan_annual_rate/12
   - payment = P * r / (1 - (1+r)^(-n))
   - interest[t] = balance[t-1]*r
   - principal_paid[t] = payment - interest[t]
   - balance[t] = max(0, balance[t-1] - principal_paid[t])
   - debt_payment[t] = payment while balance > 0 else 0

   Line of credit (LOC) (if enabled):
   - If computed ending cash before LOC < min_cash_target, draw LOC up to loc_limit to bring cash up toward min_cash_target.
   - If computed ending cash after operations and financing exceeds (min_cash_target + loc_repay_buffer), repay LOC as much as possible.
   - Interest on LOC accrues monthly on outstanding balance at loc_annual_rate/12 and is included in financing costs.

7) Capex
   - New trucks when techs increase (only if capex_trucks_mode is purchase_with_downpayment):
     new_trucks = max(trucks[t] - trucks[t-1], 0)
     capex_trucks = new_trucks * truck_purchase_price * truck_downpayment_pct
   - Tools capex for new techs:
     new_techs = max(techs[t] - techs[t-1], 0)
     tools_capex = new_techs * tools_per_new_tech_capex
   - Total capex = capex_trucks + tools_capex

8) Owner distributions
   - If enabled:
     distributions[t] = distributions_pct_of_ebitda * max(0, EBITDA[t]) only if ending cash before distributions exceeds distributions_only_if_cash_above
   - Show distributions as a financing cash outflow.

9) Cash flow statement (monthly line-by-line)
   Section A Operating
     EBITDA
     - Change in NWC
     = Operating Cash Flow

   Section B Investing
     - Capex
     = Free Cash Flow (pre-financing)

   Section C Financing
     - Debt payment (split interest and principal columns)
     +/- LOC draw or repay
     - Owner distributions
     = Net financing cash flow

   Net Cash Flow = Operating Cash Flow - Capex + Financing Cash Flow
   Begin Cash, End Cash roll forward from starting_cash.

OUTPUT TABLE REQUIREMENTS
Provide one monthly dataframe (length horizon_months) with these columns:
- Year, Month_Number, Date, Year_Month_Label
- Techs
- Calls
- Service Revenue
- Replacement Leads
- Replacement Jobs
- Replacement Revenue
- Maintenance Agreements
- Maintenance Revenue
- Total Revenue
- Financing Fee Cost
- Service Materials
- Replacement Equipment
- Permits
- Disposal
- Direct Labor
- Maintenance Direct Cost
- Total Direct Costs
- Gross Profit
- Fixed OPEX
- Marketing Spend
- Fleet Cost
- Total OPEX
- EBITDA
- AR Balance
- AP Balance
- Inventory Balance
- NWC
- Change in NWC
- Operating Cash Flow
- Capex
- Free Cash Flow
- Term Loan Payment
- Term Loan Interest
- Term Loan Principal
- Term Loan Balance
- LOC Draw
- LOC Repay
- LOC Interest
- LOC Balance
- Owner Distributions
- Net Cash Flow
- Begin Cash
- End Cash

METRICS (DISPLAY IN DASHBOARD)
Compute and display:
- Total Revenue by year (Years 1 to 5, and up to last full year if horizon_months differs)
- EBITDA by year and EBITDA margin by year
- Free Cash Flow by year
- Gross margin by year and full-period average
- Revenue per tech per year (annualized) and per truck
- CAC (Customer Acquisition Cost):
  CAC = Marketing Spend / New Customers
  Define New Customers proxy as: (Calls * attach_rate) + Replacement Jobs
- Break-even revenue:
  Break Even Revenue = Fixed Costs / Contribution Margin %
  where contribution margin % = (Total Revenue - Total Direct Costs) / Total Revenue
- Cash conversion cycle (CCC) = ar_days + inventory_days - ap_days
- DSCR (Debt Service Coverage Ratio):
  DSCR = EBITDA / (Term Loan Payment + LOC Interest) annualized or year-level
- Minimum ending cash over the full horizon and the month it occurs
- Count of months with negative ending cash
- Revenue CAGR and EBITDA CAGR over the last full year versus year 1 when horizon >= 24 months:
  CAGR = (Value_last_year / Value_year1)^(1/(years-1)) - 1

CHARTS (MINIMUM)
- Revenue by segment (stacked area)
- EBITDA and EBITDA margin (line, dual axis if using plotly)
- Ending cash balance (line)
- Gross margin percent and OPEX percent of revenue (lines)
- Optional: monthly waterfall for a selected month (Revenue -> GP -> EBITDA -> OCF -> FCF -> Net CF)
Provide a view filter for charts:
- Full horizon
- Year 1 only
- Last 12 months
- Rolling 24 months (if horizon_months >= 24)

SENSITIVITY ANALYSIS (ONE-WAY)
- Sensitivity delta percent: user adjustable (default +/-10%).
- Run one-way sensitivity on these drivers at minimum:
  avg_service_ticket
  calls_per_tech_per_day
  repl_close_rate
  avg_repl_ticket
  repl_equipment_pct
  tech_wage_per_hour
  cost_per_lead
  ar_days
- For each variable, compute low and high case holding all else constant.
- Report impacts on:
  - Year 1 EBITDA
  - Year 1 Free Cash Flow
  - Year 5 EBITDA (or last full year if horizon_months < 60)
  - Year 5 Free Cash Flow (or last full year if horizon_months < 60)
  - Minimum Ending Cash across full horizon
- Show a tornado chart for the selected target metric:
  default to Year 5 EBITDA if horizon_months >= 60 else last full year EBITDA.

IMPLEMENTATION DETAILS
- Use a single structured schema for inputs:
  either a pydantic BaseModel or a dataclass.
- Validate inputs:
  - percentages in [0,1]
  - days non-negative
  - tech counts non-negative
  - rates non-negative
- Separate engine from UI.
- Ensure deterministic results.
- Provide inline comments explaining each major block and each line item in cash flow.
- Performance:
  - Sensitivity will re-run the model 2*N times; keep it efficient for up to 120 months.

DELIVERABLE INSTRUCTIONS
- Output the complete file contents for each created file with clear file headers.
- Ensure the app runs with:
  streamlit run app.py
- Include requirements.txt pinned to reasonable versions.
- Include basic setup steps (pip install -r requirements.txt).

NOW BUILD THE CODEBASE AND OUTPUT ALL FILES.
```
