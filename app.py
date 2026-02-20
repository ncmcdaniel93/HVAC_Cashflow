import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.defaults import DEFAULTS
from src.metrics import compute_metrics
from src.model import run_model
from src.sensitivity import SENSITIVITY_DRIVERS, TARGET_OPTIONS, run_one_way_sensitivity

st.set_page_config(page_title="HVAC Cash Flow Model", layout="wide")
st.title("HVAC Cash Flow Model")
st.caption("Single scenario, driver based forecast with monthly cash flow and sensitivity analysis.")


def pct_input(label, key, help_text):
    return st.sidebar.slider(label, 0.0, 1.0, float(DEFAULTS[key]), 0.001, help=help_text)


with st.sidebar:
    st.header("Scenario Manager")
    st.subheader("Model Controls")
    start_year = st.number_input("Start Year", min_value=2000, max_value=2100, value=DEFAULTS["start_year"], help="Forecast start year.")
    start_month = st.number_input("Start Month", min_value=1, max_value=12, value=DEFAULTS["start_month"], help="Forecast start month number.")
    horizon_months = st.slider("Horizon Months", 12, 120, DEFAULTS["horizon_months"], help="Forecast length in months.")
    monthly_price_growth = pct_input("Monthly Price Growth", "monthly_price_growth", "Monthly increase applied to revenue price drivers.")
    monthly_cost_inflation = pct_input("Monthly Cost Inflation", "monthly_cost_inflation", "Monthly increase applied to cost drivers.")
    seasonality_amplitude = pct_input("Seasonality Amplitude", "seasonality_amplitude", "Sinusoid amplitude for seasonality.")
    peak_month = st.number_input("Peak Month", min_value=1, max_value=12, value=DEFAULTS["peak_month"], help="Calendar month where seasonality peaks.")

    st.subheader("Staffing and Ops")
    starting_techs = st.number_input("Starting Techs", min_value=0, value=DEFAULTS["starting_techs"], help="Technicians at month 1.")
    tech_hire_per_quarter = st.number_input("Tech Hire per Quarter", min_value=0, value=DEFAULTS["tech_hire_per_quarter"], help="Technicians added every three months.")
    max_techs = st.number_input("Max Techs", min_value=0, value=DEFAULTS["max_techs"], help="Upper cap on technicians.")
    calls_per_tech_per_day = st.number_input("Calls per Tech per Day", min_value=0.0, value=DEFAULTS["calls_per_tech_per_day"], help="Average service calls completed by each technician per day.")
    work_days_per_month = st.number_input("Work Days per Month", min_value=1.0, value=float(DEFAULTS["work_days_per_month"]), help="Working days used to convert daily calls to monthly calls.")
    avg_hours_per_tech_per_month = st.number_input("Avg Hours per Tech per Month", min_value=0.0, value=float(DEFAULTS["avg_hours_per_tech_per_month"]), help="Paid labor hours per technician each month.")
    tech_wage_per_hour = st.number_input("Tech Wage per Hour", min_value=0.0, value=DEFAULTS["tech_wage_per_hour"], help="Hourly wage for technicians.")
    payroll_burden_pct = pct_input("Payroll Burden Pct", "payroll_burden_pct", "Payroll taxes and benefits as a percent of wages.")
    tools_per_new_tech_capex = st.number_input("Tools per New Tech Capex", min_value=0.0, value=DEFAULTS["tools_per_new_tech_capex"], help="Upfront tools investment per newly hired technician.")

    st.subheader("Service Revenue")
    avg_service_ticket = st.number_input("Avg Service Ticket", min_value=0.0, value=DEFAULTS["avg_service_ticket"], help="Average revenue per service call.")
    service_material_pct = pct_input("Service Material Pct", "service_material_pct", "Material cost as a percent of service revenue.")
    attach_rate = pct_input("Attach Rate", "attach_rate", "Proxy conversion from calls to new customers.")

    st.subheader("Replacement Revenue")
    repl_leads_per_tech_per_month = st.number_input("Replacement Leads per Tech per Month", min_value=0.0, value=DEFAULTS["repl_leads_per_tech_per_month"], help="Replacement opportunities generated per technician monthly.")
    repl_close_rate = pct_input("Replacement Close Rate", "repl_close_rate", "Close rate on replacement leads.")
    avg_repl_ticket = st.number_input("Avg Replacement Ticket", min_value=0.0, value=DEFAULTS["avg_repl_ticket"], help="Average replacement job revenue.")
    repl_equipment_pct = pct_input("Replacement Equipment Pct", "repl_equipment_pct", "Equipment cost as a percent of replacement revenue.")
    permit_cost_per_repl_job = st.number_input("Permit Cost per Replacement Job", min_value=0.0, value=DEFAULTS["permit_cost_per_repl_job"], help="Permit cost per replacement job.")
    disposal_cost_per_repl_job = st.number_input("Disposal Cost per Replacement Job", min_value=0.0, value=DEFAULTS["disposal_cost_per_repl_job"], help="Disposal cost per replacement job.")
    financing_penetration = pct_input("Financing Penetration", "financing_penetration", "Share of replacement revenue financed by customer financing.")
    financing_fee_pct = pct_input("Financing Fee Pct", "financing_fee_pct", "Merchant financing fee as a percent of financed replacement revenue.")

    st.subheader("Maintenance Program")
    enable_maintenance = st.checkbox("Enable Maintenance", value=DEFAULTS["enable_maintenance"], help="Toggle maintenance agreements.")
    agreements_start = st.number_input("Agreements Start", min_value=0.0, value=float(DEFAULTS["agreements_start"]), help="Initial active maintenance agreements.")
    new_agreements_per_month = st.number_input("New Agreements per Month", min_value=0.0, value=float(DEFAULTS["new_agreements_per_month"]), help="Net new maintenance agreements per month before churn effects.")
    churn_annual_pct = pct_input("Churn Annual Pct", "churn_annual_pct", "Annual churn rate for maintenance agreements.")
    maint_monthly_fee = st.number_input("Maintenance Monthly Fee", min_value=0.0, value=DEFAULTS["maint_monthly_fee"], help="Monthly maintenance agreement fee.")
    cost_per_maint_visit = st.number_input("Cost per Maintenance Visit", min_value=0.0, value=DEFAULTS["cost_per_maint_visit"], help="Direct cost per maintenance visit.")
    maint_visits_per_agreement_per_year = st.number_input("Maintenance Visits per Agreement per Year", min_value=0.0, value=DEFAULTS["maint_visits_per_agreement_per_year"], help="Expected annual visits per maintenance agreement.")

    st.subheader("Marketing")
    paid_leads_mode = st.selectbox("Paid Leads Mode", ["fixed", "per_tech"], index=1, help="Choose fixed paid leads per month or leads that scale with technicians.")
    paid_leads_per_month = st.number_input("Paid Leads per Month", min_value=0.0, value=float(DEFAULTS["paid_leads_per_month"]), help="Used when paid leads mode is fixed.")
    paid_leads_per_tech_per_month = st.number_input("Paid Leads per Tech per Month", min_value=0.0, value=float(DEFAULTS["paid_leads_per_tech_per_month"]), help="Used when paid leads mode is per technician.")
    cost_per_lead = st.number_input("Cost per Lead", min_value=0.0, value=DEFAULTS["cost_per_lead"], help="Paid marketing cost per lead.")
    branding_fixed_monthly = st.number_input("Branding Fixed Monthly", min_value=0.0, value=DEFAULTS["branding_fixed_monthly"], help="Fixed monthly branding spend.")

    st.subheader("Fleet")
    trucks_per_tech = st.number_input("Trucks per Tech", min_value=0.0, value=DEFAULTS["trucks_per_tech"], help="Truck to technician ratio.")
    truck_payment_monthly = st.number_input("Truck Payment Monthly", min_value=0.0, value=DEFAULTS["truck_payment_monthly"], help="Monthly truck payment.")
    fuel_per_truck_monthly = st.number_input("Fuel per Truck Monthly", min_value=0.0, value=DEFAULTS["fuel_per_truck_monthly"], help="Monthly fuel per truck.")
    maint_per_truck_monthly = st.number_input("Maintenance per Truck Monthly", min_value=0.0, value=DEFAULTS["maint_per_truck_monthly"], help="Monthly maintenance per truck.")
    truck_insurance_per_truck_monthly = st.number_input("Truck Insurance per Truck Monthly", min_value=0.0, value=DEFAULTS["truck_insurance_per_truck_monthly"], help="Monthly insurance per truck.")
    truck_purchase_price = st.number_input("Truck Purchase Price", min_value=0.0, value=DEFAULTS["truck_purchase_price"], help="Purchase price per truck.")
    capex_trucks_mode = st.selectbox("Capex Trucks Mode", ["payments_only", "purchase_with_downpayment"], index=1, help="Payments only ignores downpayment capex on new trucks.")
    truck_downpayment_pct = pct_input("Truck Downpayment Pct", "truck_downpayment_pct", "Downpayment percent for each new truck purchase.")
    truck_financed_pct = pct_input("Truck Financed Pct", "truck_financed_pct", "Financed portion of truck purchase.")

    st.subheader("Overhead")
    office_payroll_monthly = st.number_input("Office Payroll Monthly", min_value=0.0, value=DEFAULTS["office_payroll_monthly"])
    rent_monthly = st.number_input("Rent Monthly", min_value=0.0, value=DEFAULTS["rent_monthly"])
    utilities_monthly = st.number_input("Utilities Monthly", min_value=0.0, value=DEFAULTS["utilities_monthly"])
    insurance_monthly = st.number_input("Insurance Monthly", min_value=0.0, value=DEFAULTS["insurance_monthly"])
    software_monthly = st.number_input("Software Monthly", min_value=0.0, value=DEFAULTS["software_monthly"])
    other_fixed_monthly = st.number_input("Other Fixed Monthly", min_value=0.0, value=DEFAULTS["other_fixed_monthly"])

    st.subheader("Working Capital")
    ar_days = st.number_input("AR Days", min_value=0.0, value=DEFAULTS["ar_days"], help="Average days sales outstanding.")
    ap_days = st.number_input("AP Days", min_value=0.0, value=DEFAULTS["ap_days"], help="Average days payable outstanding.")
    inventory_days = st.number_input("Inventory Days", min_value=0.0, value=DEFAULTS["inventory_days"], help="Average days inventory on hand.")
    starting_cash = st.number_input("Starting Cash", min_value=0.0, value=DEFAULTS["starting_cash"], help="Beginning cash balance in month 1.")

    st.subheader("Debt")
    enable_term_loan = st.checkbox("Enable Term Loan", value=DEFAULTS["enable_term_loan"])
    loan_principal = st.number_input("Loan Principal", min_value=0.0, value=DEFAULTS["loan_principal"])
    loan_annual_rate = pct_input("Loan Annual Rate", "loan_annual_rate", "Annual term loan interest rate.")
    loan_term_months = st.number_input("Loan Term Months", min_value=1, value=DEFAULTS["loan_term_months"])
    enable_loc = st.checkbox("Enable LOC", value=DEFAULTS["enable_loc"], help="LOC means line of credit.")
    loc_limit = st.number_input("LOC Limit", min_value=0.0, value=DEFAULTS["loc_limit"])
    loc_annual_rate = pct_input("LOC Annual Rate", "loc_annual_rate", "Annual interest rate on the line of credit.")
    min_cash_target = st.number_input("Min Cash Target", min_value=0.0, value=DEFAULTS["min_cash_target"])
    loc_repay_buffer = st.number_input("LOC Repay Buffer", min_value=0.0, value=DEFAULTS["loc_repay_buffer"])

    st.subheader("Distributions")
    enable_distributions = st.checkbox("Enable Distributions", value=DEFAULTS["enable_distributions"])
    distributions_pct_of_ebitda = pct_input("Distributions Pct of EBITDA", "distributions_pct_of_ebitda", "Share of positive EBITDA distributed to owners.")
    distributions_only_if_cash_above = st.number_input("Distributions Only if Cash Above", min_value=0.0, value=DEFAULTS["distributions_only_if_cash_above"])

    sensitivity_delta = st.slider("Sensitivity Delta Percent", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

inputs = {
    k: v
    for k, v in locals().items()
    if k in DEFAULTS
}

df = run_model(inputs)
df.attrs["attach_rate"] = inputs["attach_rate"]
df.attrs["ar_days"] = inputs["ar_days"]
df.attrs["ap_days"] = inputs["ap_days"]
df.attrs["inventory_days"] = inputs["inventory_days"]

metrics = compute_metrics(df, horizon_months)

chart_filter = st.selectbox("Chart View", ["Full horizon", "Year 1 only", "Last 12 months", "Rolling 24 months"]) 
if chart_filter == "Year 1 only":
    chart_df = df[df["Year"] == 1]
elif chart_filter == "Last 12 months":
    chart_df = df.tail(12)
elif chart_filter == "Rolling 24 months" and len(df) >= 24:
    chart_df = df.tail(24)
else:
    chart_df = df

summary_tab, cashflow_tab, drivers_tab, sens_tab = st.tabs([
    "Summary Dashboard",
    "Line-by-line Cash Flow",
    "Drivers",
    "Sensitivity",
])

with summary_tab:
    st.subheader("Headline KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${df['Total Revenue'].sum():,.0f}")
    c2.metric("Total EBITDA", f"${df['EBITDA'].sum():,.0f}")
    c3.metric("Total Free Cash Flow", f"${df['Free Cash Flow'].sum():,.0f}")
    c4.metric("Minimum Ending Cash", f"${metrics['minimum_ending_cash']:,.0f}", metrics["minimum_ending_cash_month"])

    st.write("EBITDA means Earnings Before Interest, Taxes, Depreciation, and Amortization.")
    st.write(f"Cash Conversion Cycle: {metrics['ccc']:.1f} days")
    st.write(f"CAC (Customer Acquisition Cost): ${metrics['cac']:,.2f}")
    st.write(f"Break-even revenue: ${metrics['break_even_revenue']:,.0f}")

    seg = chart_df[["Date", "Service Revenue", "Replacement Revenue", "Maintenance Revenue"]].melt("Date", var_name="Segment", value_name="Revenue")
    st.plotly_chart(px.area(seg, x="Date", y="Revenue", color="Segment", title="Revenue by Segment"), use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df["Date"], y=chart_df["EBITDA"], name="EBITDA"))
    fig.add_trace(go.Scatter(x=chart_df["Date"], y=100 * chart_df["EBITDA"] / chart_df["Total Revenue"].replace(0, pd.NA), name="EBITDA Margin %", yaxis="y2"))
    fig.update_layout(title="EBITDA and EBITDA Margin", yaxis2=dict(overlaying="y", side="right"))
    st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(px.line(chart_df, x="Date", y="End Cash", title="Ending Cash Balance"), use_container_width=True)

    gm = 100 * chart_df["Gross Profit"] / chart_df["Total Revenue"].replace(0, pd.NA)
    op = 100 * chart_df["Total OPEX"] / chart_df["Total Revenue"].replace(0, pd.NA)
    gm_df = pd.DataFrame({"Date": chart_df["Date"], "Gross Margin %": gm, "OPEX % of Revenue": op})
    gm_melt = gm_df.melt("Date", var_name="Metric", value_name="Percent")
    st.plotly_chart(px.line(gm_melt, x="Date", y="Percent", color="Metric", title="Gross Margin % and OPEX % of Revenue"), use_container_width=True)

with cashflow_tab:
    st.subheader("Monthly line-by-line cash flow")
    st.write("Operating Cash Flow equals EBITDA minus change in net working capital.")
    st.write("Free Cash Flow equals Operating Cash Flow minus capex.")
    st.write("Net Cash Flow equals Operating Cash Flow minus capex plus net financing cash flow.")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), file_name="hvac_cashflow_monthly.csv", mime="text/csv")

with drivers_tab:
    st.subheader("Key drivers over time")
    drv = df[["Date", "Techs", "Calls", "Replacement Leads", "Service Revenue", "Replacement Revenue", "Maintenance Revenue"]]
    st.plotly_chart(px.line(drv, x="Date", y=["Techs", "Calls", "Replacement Leads"], title="Ops Drivers"), use_container_width=True)
    st.plotly_chart(px.line(drv, x="Date", y=["Service Revenue", "Replacement Revenue", "Maintenance Revenue"], title="Revenue Drivers"), use_container_width=True)

with sens_tab:
    st.subheader("One-way sensitivity")
    sens_df, target_year = run_one_way_sensitivity(inputs, sensitivity_delta)
    default_target = "Year N EBITDA" if horizon_months >= 60 else "Year N EBITDA"
    target = st.selectbox("Target metric", TARGET_OPTIONS, index=TARGET_OPTIONS.index(default_target))

    tornado = sens_df.pivot(index="Driver", columns="Case", values=f"Delta {target}").fillna(0)
    tornado = tornado.reindex(SENSITIVITY_DRIVERS)
    tornado["Base"] = 0
    tdf = tornado[["Low", "Base", "High"]].reset_index().melt(id_vars="Driver", var_name="Case", value_name="Delta")
    st.plotly_chart(px.bar(tdf, x="Delta", y="Driver", color="Case", orientation="h", title=f"Tornado Chart for {target} (Year N = {target_year})"), use_container_width=True)
    st.dataframe(sens_df, use_container_width=True)
