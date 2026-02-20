"""Model accounting and roll-forward integrity checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _finding(
    check: str,
    max_abs_delta: float,
    month: str,
    lhs_name: str,
    rhs_name: str,
) -> dict[str, Any]:
    return {
        "Check": check,
        "Max Abs Delta": float(max_abs_delta),
        "Month of Max Delta": month,
        "LHS": lhs_name,
        "RHS": rhs_name,
    }


def _month_of_max_delta(df: pd.DataFrame, delta: np.ndarray) -> str:
    if len(delta) == 0:
        return ""
    idx = int(np.argmax(np.abs(delta)))
    if "Year_Month_Label" in df.columns and idx < len(df):
        return str(df.iloc[idx]["Year_Month_Label"])
    return str(idx)


def _check_series_identity(
    findings: list[dict[str, Any]],
    df: pd.DataFrame,
    check_name: str,
    lhs_name: str,
    rhs_name: str,
    lhs: np.ndarray,
    rhs: np.ndarray,
    tol: float,
) -> None:
    delta = np.nan_to_num(np.asarray(lhs, dtype=float) - np.asarray(rhs, dtype=float), nan=0.0)
    if len(delta) == 0:
        return
    max_abs = float(np.max(np.abs(delta)))
    if max_abs > float(tol):
        findings.append(_finding(check_name, max_abs, _month_of_max_delta(df, delta), lhs_name, rhs_name))


def run_integrity_checks(df: pd.DataFrame, assumptions: dict, tol: float = 1e-3) -> list[dict[str, Any]]:
    """Return integrity findings (empty list means all checks passed)."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return [{"Check": "Dataframe not available", "Max Abs Delta": np.nan, "Month of Max Delta": "", "LHS": "", "RHS": ""}]

    findings: list[dict[str, Any]] = []

    # Core P&L identities.
    _check_series_identity(
        findings,
        df,
        "Revenue identity",
        "Total Revenue",
        "Service+Replacement+Maintenance+Upsell+New Build",
        df["Total Revenue"].to_numpy(),
        (
            df["Service Revenue"]
            + df["Replacement Revenue"]
            + df["Maintenance Revenue"]
            + df["Upsell Revenue"]
            + df["New Build Revenue"]
        ).to_numpy(),
        tol,
    )
    _check_series_identity(
        findings,
        df,
        "Direct cost identity",
        "Total Direct Costs",
        "Direct cost components",
        df["Total Direct Costs"].to_numpy(),
        (
            df["Service Materials"]
            + df["Replacement Equipment"]
            + df["Permits"]
            + df["Disposal"]
            + df["Direct Labor"]
            + df["Maintenance Direct Cost"]
            + df["Financing Fee Cost"]
            + df["Upsell Direct Cost"]
            + df["New Build Direct Cost"]
        ).to_numpy(),
        tol,
    )
    _check_series_identity(
        findings,
        df,
        "Gross profit identity",
        "Gross Profit",
        "Total Revenue - Total Direct Costs",
        df["Gross Profit"].to_numpy(),
        (df["Total Revenue"] - df["Total Direct Costs"]).to_numpy(),
        tol,
    )
    _check_series_identity(
        findings,
        df,
        "OPEX identity",
        "Total OPEX",
        "Fixed+Marketing+Fleet+Sales+Management payroll",
        df["Total OPEX"].to_numpy(),
        (df["Fixed OPEX"] + df["Marketing Spend"] + df["Fleet Cost"] + df["Sales Payroll"] + df["Management Payroll"]).to_numpy(),
        tol,
    )
    _check_series_identity(
        findings,
        df,
        "EBITDA identity",
        "EBITDA",
        "Gross Profit - Total OPEX",
        df["EBITDA"].to_numpy(),
        (df["Gross Profit"] - df["Total OPEX"]).to_numpy(),
        tol,
    )

    # Working capital and cash flow identities.
    _check_series_identity(
        findings,
        df,
        "NWC identity",
        "NWC",
        "AR + Inventory - AP",
        df["NWC"].to_numpy(),
        (df["AR Balance"] + df["Inventory Balance"] - df["AP Balance"]).to_numpy(),
        tol,
    )
    if len(df) > 1:
        _check_series_identity(
            findings,
            df.iloc[1:].reset_index(drop=True),
            "Change in NWC roll-forward",
            "Change in NWC",
            "NWC[t] - NWC[t-1]",
            df["Change in NWC"].to_numpy()[1:],
            np.diff(df["NWC"].to_numpy()),
            tol,
        )
    _check_series_identity(
        findings,
        df,
        "Operating cash flow identity",
        "Operating Cash Flow",
        "EBITDA - Change in NWC",
        df["Operating Cash Flow"].to_numpy(),
        (df["EBITDA"] - df["Change in NWC"]).to_numpy(),
        tol,
    )
    _check_series_identity(
        findings,
        df,
        "Gross capex identity",
        "Gross Capex",
        "Tools Capex + Truck Capex",
        df["Gross Capex"].to_numpy(),
        (df["Tools Capex"] + df["Truck Capex"]).to_numpy(),
        tol,
    )
    _check_series_identity(
        findings,
        df,
        "Net capex identity",
        "Capex",
        "Gross Capex - Asset Salvage Proceeds",
        df["Capex"].to_numpy(),
        (df["Gross Capex"] - df["Asset Salvage Proceeds"]).to_numpy(),
        tol,
    )
    _check_series_identity(
        findings,
        df,
        "Free cash flow identity",
        "Free Cash Flow",
        "Operating Cash Flow - Capex",
        df["Free Cash Flow"].to_numpy(),
        (df["Operating Cash Flow"] - df["Capex"]).to_numpy(),
        tol,
    )
    _check_series_identity(
        findings,
        df,
        "Net financing cash flow identity",
        "Net Financing Cash Flow",
        "-Loan Pmt - LOC Int + LOC Draw - LOC Repay - Distributions",
        df["Net Financing Cash Flow"].to_numpy(),
        (-df["Term Loan Payment"] - df["LOC Interest"] + df["LOC Draw"] - df["LOC Repay"] - df["Owner Distributions"]).to_numpy(),
        tol,
    )
    _check_series_identity(
        findings,
        df,
        "Net cash flow identity",
        "Net Cash Flow",
        "Operating Cash Flow - Capex + Net Financing Cash Flow",
        df["Net Cash Flow"].to_numpy(),
        (df["Operating Cash Flow"] - df["Capex"] + df["Net Financing Cash Flow"]).to_numpy(),
        tol,
    )
    _check_series_identity(
        findings,
        df,
        "Cash roll-forward",
        "End Cash",
        "Begin Cash + Net Cash Flow",
        df["End Cash"].to_numpy(),
        (df["Begin Cash"] + df["Net Cash Flow"]).to_numpy(),
        tol,
    )

    # Debt roll-forwards.
    _check_series_identity(
        findings,
        df,
        "Term loan payment split",
        "Term Loan Payment",
        "Term Loan Interest + Term Loan Principal",
        df["Term Loan Payment"].to_numpy(),
        (df["Term Loan Interest"] + df["Term Loan Principal"]).to_numpy(),
        tol,
    )
    opening_loan_balance = float(assumptions.get("loan_principal", 0.0)) if bool(assumptions.get("enable_term_loan", False)) else 0.0
    prev_loan_bal = np.concatenate(([opening_loan_balance], df["Term Loan Balance"].to_numpy()[:-1]))
    _check_series_identity(
        findings,
        df,
        "Term loan balance roll-forward",
        "Term Loan Balance",
        "Prior Balance - Term Loan Principal",
        df["Term Loan Balance"].to_numpy(),
        np.maximum(0.0, prev_loan_bal - df["Term Loan Principal"].to_numpy()),
        tol,
    )
    prev_loc_bal = np.concatenate(([0.0], df["LOC Balance"].to_numpy()[:-1]))
    _check_series_identity(
        findings,
        df,
        "LOC balance roll-forward",
        "LOC Balance",
        "Prior Balance + LOC Draw - LOC Repay",
        df["LOC Balance"].to_numpy(),
        np.maximum(0.0, prev_loc_bal + df["LOC Draw"].to_numpy() - df["LOC Repay"].to_numpy()),
        tol,
    )

    return findings
