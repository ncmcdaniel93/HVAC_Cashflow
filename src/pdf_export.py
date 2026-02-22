"""Analyst-grade PDF export helpers for the current scenario report."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import html
from io import BytesIO
import subprocess
import sys
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.input_metadata import INPUT_GUIDANCE


DEFAULT_OPTIONS = {
    "chart_pack": "comprehensive",
    "include_enabled_custom_charts": True,
    "appendix_detail": "maximum",
    "time_basis": "range_plus_full_reference",
    "chart_width_px": 1400,
    "chart_height_px": 800,
    "allow_dependency_bootstrap": False,
    "dependency_bootstrap_timeout_sec": 240,
}

LOWER_IS_BETTER_TARGETS = {
    "Total Disbursements",
    "Break-even Revenue",
    "Break-even Labor Rate",
    "Break-even Wage Rate",
}

_KALEIDO_READY: bool | None = None
_REPORTLAB_READY: bool | None = None


def _merge_options(options: dict | None) -> dict:
    out = deepcopy(DEFAULT_OPTIONS)
    if isinstance(options, dict):
        out.update(options)
    return out


def _log_event(
    options: dict,
    *,
    level: str,
    event: str,
    message: str,
    context: dict[str, Any] | None = None,
    exc: BaseException | None = None,
) -> None:
    logger: Callable[..., Any] | None = options.get("log_event")
    if not callable(logger):
        return
    try:
        logger(level=level, event=event, message=message, context=context or {}, exc=exc)
    except Exception:
        # Export logging should never break report generation.
        return


def _run_pip_install(packages: list[str], timeout_sec: int) -> tuple[bool, str]:
    if not packages:
        return True, ""
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        *packages,
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(30, int(timeout_sec)),
        )
    except Exception as exc:
        return False, str(exc)
    if completed.returncode == 0:
        return True, completed.stdout.strip()
    err = (completed.stderr or completed.stdout or "").strip()
    return False, err


def _ensure_reportlab_ready(options: dict) -> bool:
    global _REPORTLAB_READY
    if _REPORTLAB_READY is True:
        return True
    try:
        import reportlab  # noqa: F401

        _REPORTLAB_READY = True
        return True
    except Exception as initial_exc:
        if not bool(options.get("allow_dependency_bootstrap", False)):
            _REPORTLAB_READY = False
            return False
        timeout_sec = _safe_int(options.get("dependency_bootstrap_timeout_sec", 240), 240)
        _log_event(
            options,
            level="INFO",
            event="pdf_dependency_install_started",
            message="ReportLab is missing; attempting runtime install.",
            context={"package": "reportlab"},
            exc=initial_exc,
        )
        ok, detail = _run_pip_install(["reportlab>=4.4.1"], timeout_sec=timeout_sec)
        if not ok:
            _REPORTLAB_READY = False
            _log_event(
                options,
                level="ERROR",
                event="pdf_dependency_install_failed",
                message="Failed to install ReportLab for PDF export.",
                context={"package": "reportlab", "detail": detail},
            )
            return False
        try:
            import reportlab  # noqa: F401

            _REPORTLAB_READY = True
            _log_event(
                options,
                level="INFO",
                event="pdf_dependency_install_succeeded",
                message="ReportLab installed successfully for PDF export.",
                context={"package": "reportlab"},
            )
            return True
        except Exception as exc:
            _REPORTLAB_READY = False
            _log_event(
                options,
                level="ERROR",
                event="pdf_dependency_install_failed",
                message="ReportLab install completed but import still failed.",
                context={"package": "reportlab", "detail": detail},
                exc=exc,
            )
            return False


def _ensure_kaleido_ready(options: dict) -> bool:
    global _KALEIDO_READY
    if _KALEIDO_READY is True:
        return True
    timeout_sec = _safe_int(options.get("dependency_bootstrap_timeout_sec", 240), 240)
    allow_bootstrap = bool(options.get("allow_dependency_bootstrap", False))

    def _smoke_test() -> bool:
        fig = go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])])
        fig.update_layout(width=500, height=300, template="plotly_white")
        fig.to_image(format="png", width=500, height=300, scale=1)
        return True

    try:
        import kaleido  # noqa: F401

        try:
            _smoke_test()
            _KALEIDO_READY = True
            return True
        except Exception as smoke_exc:
            if not allow_bootstrap:
                _KALEIDO_READY = False
                _log_event(
                    options,
                    level="WARNING",
                    event="pdf_chart_engine_unavailable",
                    message="Kaleido import succeeded but chart export smoke test failed.",
                    context={"detail": str(smoke_exc)},
                )
                return False
            _log_event(
                options,
                level="INFO",
                event="pdf_chart_engine_bootstrap_started",
                message="Kaleido smoke test failed; attempting Chrome bootstrap for chart export.",
                context={},
                exc=smoke_exc,
            )
            try:
                import plotly.io as pio

                if hasattr(pio, "get_chrome"):
                    pio.get_chrome()
            except Exception as chrome_exc:
                _KALEIDO_READY = False
                _log_event(
                    options,
                    level="ERROR",
                    event="pdf_chart_engine_bootstrap_failed",
                    message="Failed to bootstrap Chrome for Kaleido chart export.",
                    context={},
                    exc=chrome_exc,
                )
                return False
            try:
                _smoke_test()
                _KALEIDO_READY = True
                _log_event(
                    options,
                    level="INFO",
                    event="pdf_chart_engine_bootstrap_succeeded",
                    message="Kaleido chart engine ready after Chrome bootstrap.",
                    context={},
                )
                return True
            except Exception as final_exc:
                _KALEIDO_READY = False
                _log_event(
                    options,
                    level="ERROR",
                    event="pdf_chart_engine_bootstrap_failed",
                    message="Kaleido chart engine remained unavailable after bootstrap.",
                    context={},
                    exc=final_exc,
                )
                return False
    except Exception as import_exc:
        if not allow_bootstrap:
            _KALEIDO_READY = False
            return False
        _log_event(
            options,
            level="INFO",
            event="pdf_dependency_install_started",
            message="Kaleido is missing; attempting runtime install.",
            context={"package": "kaleido"},
            exc=import_exc,
        )
        ok, detail = _run_pip_install(["kaleido>=1.2.0"], timeout_sec=timeout_sec)
        if not ok:
            _KALEIDO_READY = False
            _log_event(
                options,
                level="ERROR",
                event="pdf_dependency_install_failed",
                message="Failed to install Kaleido for chart export.",
                context={"package": "kaleido", "detail": detail},
            )
            return False
        try:
            import kaleido  # noqa: F401

            try:
                _smoke_test()
                _KALEIDO_READY = True
                _log_event(
                    options,
                    level="INFO",
                    event="pdf_dependency_install_succeeded",
                    message="Kaleido installed and ready.",
                    context={"package": "kaleido"},
                )
                return True
            except Exception:
                # Try Chrome bootstrap once after successful install.
                import plotly.io as pio

                if hasattr(pio, "get_chrome"):
                    pio.get_chrome()
                _smoke_test()
                _KALEIDO_READY = True
                _log_event(
                    options,
                    level="INFO",
                    event="pdf_chart_engine_bootstrap_succeeded",
                    message="Kaleido installed and ready after Chrome bootstrap.",
                    context={},
                )
                return True
        except Exception as exc:
            _KALEIDO_READY = False
            _log_event(
                options,
                level="ERROR",
                event="pdf_dependency_install_failed",
                message="Kaleido install completed but chart engine remains unavailable.",
                context={"package": "kaleido", "detail": detail},
                exc=exc,
            )
            return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _is_dataframe(value: Any) -> bool:
    return isinstance(value, pd.DataFrame)


def _ensure_dataframe(value: Any) -> pd.DataFrame:
    if _is_dataframe(value):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    return pd.DataFrame()


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_currency(value: Any) -> str:
    num = _safe_float(value, 0.0)
    return f"${num:,.0f}"


def _fmt_currency_2(value: Any) -> str:
    num = _safe_float(value, 0.0)
    return f"${num:,.2f}"


def _fmt_percent(value: Any) -> str:
    num = _safe_float(value, 0.0)
    if abs(num) <= 1.0:
        num *= 100.0
    return f"{num:,.1f}%"


def _fmt_number(value: Any) -> str:
    num = _safe_float(value, 0.0)
    return f"{num:,.1f}"


def _fmt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        if abs(float(value) - round(float(value))) < 1e-9:
            return f"{int(round(float(value))):,}"
        return f"{float(value):,.4f}".rstrip("0").rstrip(".")
    return str(value)


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_minimal_pdf(lines: list[str]) -> bytes:
    """Return a minimal but valid PDF when ReportLab is unavailable."""

    if not lines:
        lines = ["HVAC Analyst Report", "No content available."]
    max_lines = 48
    pages = [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)] or [[]]
    width = 612
    height = 792
    line_height = 14

    objects: dict[int, str] = {}
    next_id = 1

    catalog_id = next_id
    next_id += 1
    pages_id = next_id
    next_id += 1
    font_id = next_id
    next_id += 1

    page_ids: list[int] = []
    content_ids: list[int] = []
    for _ in pages:
        page_ids.append(next_id)
        next_id += 1
        content_ids.append(next_id)
        next_id += 1

    objects[font_id] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for idx, page_lines in enumerate(pages):
        content_cmds = ["BT", "/F1 10 Tf", f"50 {height - 50} Td", f"{line_height} TL"]
        for i, line in enumerate(page_lines):
            safe = _pdf_escape(line[:220])
            if i == 0:
                content_cmds.append(f"({safe}) Tj")
            else:
                content_cmds.append(f"T* ({safe}) Tj")
        content_cmds.append("ET")
        content_stream = "\n".join(content_cmds).encode("latin-1", errors="replace")
        content_id = content_ids[idx]
        page_id = page_ids[idx]
        objects[content_id] = (
            f"<< /Length {len(content_stream)} >>\nstream\n"
            + content_stream.decode("latin-1", errors="replace")
            + "\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        )

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_id] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>"
    objects[catalog_id] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>"

    out = BytesIO()
    out.write(b"%PDF-1.4\n")
    xref_offsets: dict[int, int] = {0: 0}
    for obj_id in range(1, next_id):
        xref_offsets[obj_id] = out.tell()
        out.write(f"{obj_id} 0 obj\n".encode("ascii"))
        out.write(objects[obj_id].encode("latin-1", errors="replace"))
        out.write(b"\nendobj\n")
    xref_start = out.tell()
    out.write(f"xref\n0 {next_id}\n".encode("ascii"))
    out.write(b"0000000000 65535 f \n")
    for obj_id in range(1, next_id):
        out.write(f"{xref_offsets[obj_id]:010d} 00000 n \n".encode("ascii"))
    out.write(
        (
            f"trailer\n<< /Size {next_id} /Root {catalog_id} 0 R >>\nstartxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )
    return out.getvalue()


def render_plotly_figure_png(fig, width_px: int, height_px: int) -> bytes:
    """Render a Plotly figure into PNG bytes using Kaleido."""

    if fig is None:
        raise ValueError("Figure is required.")
    width_px = max(640, _safe_int(width_px, 1400))
    height_px = max(360, _safe_int(height_px, 800))
    fig.update_layout(
        template="plotly_white",
        width=width_px,
        height=height_px,
        margin=dict(l=50, r=40, t=70, b=50),
    )
    try:
        image = fig.to_image(format="png", width=width_px, height=height_px, scale=1)
    except Exception as exc:
        raise RuntimeError("Chart image render failed.") from exc
    return bytes(image)


def _add_series_if_present(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    y_col: str,
    name: str,
    mode: str = "lines",
    yaxis: str | None = None,
    line_dash: str | None = None,
) -> None:
    if y_col not in df.columns:
        return
    fig.add_trace(
        go.Scatter(
            x=df["Date"] if "Date" in df.columns else list(range(len(df))),
            y=df[y_col],
            name=name,
            mode=mode,
            yaxis=yaxis,
            line=dict(dash=line_dash) if line_dash else None,
        )
    )


def _chart_placeholder(
    chart_id: str,
    title: str,
    section: str,
    reason: str,
    *,
    core: bool = True,
) -> dict[str, Any]:
    return {
        "id": chart_id,
        "title": title,
        "section": section,
        "image_bytes": None,
        "placeholder_text": reason,
        "core_chart": bool(core),
    }


def _build_fleet_cost_components(df: pd.DataFrame, assumptions: dict) -> pd.DataFrame:
    if "Fleet Cost" not in df.columns:
        return pd.DataFrame()
    trucks = pd.Series(0.0, index=df.index)
    for col in ["Trucks", "Retained Trucks"]:
        if col in df.columns:
            trucks = trucks + df[col].fillna(0.0)
    fuel = _safe_float(assumptions.get("fuel_per_truck_monthly", 0.0))
    maint = _safe_float(assumptions.get("maint_per_truck_monthly", 0.0))
    insurance = _safe_float(assumptions.get("truck_insurance_per_truck_monthly", 0.0))
    payment = _safe_float(assumptions.get("truck_payment_monthly", 0.0)) * _safe_float(
        assumptions.get("truck_financed_pct", 0.0)
    )
    unit_total = max(1e-9, fuel + maint + insurance + payment)
    ratio = {
        "Fuel": fuel / unit_total,
        "Maintenance": maint / unit_total,
        "Insurance": insurance / unit_total,
        "Truck Financing": payment / unit_total,
    }
    out = pd.DataFrame({"Date": df["Date"] if "Date" in df.columns else pd.RangeIndex(len(df))})
    for name, share in ratio.items():
        out[name] = df["Fleet Cost"] * share
    return out


def _build_sensitivity_summary_df(sens_df: pd.DataFrame, target_metric: str) -> pd.DataFrame:
    if sens_df is None or sens_df.empty:
        return pd.DataFrame()
    delta_col = f"Delta {target_metric}"
    if delta_col not in sens_df.columns:
        return pd.DataFrame()
    pivot = sens_df.pivot_table(index="Driver", columns="Case", values=delta_col, aggfunc="mean").fillna(0.0)
    if "Low" not in pivot.columns:
        pivot["Low"] = 0.0
    if "High" not in pivot.columns:
        pivot["High"] = 0.0
    sign = -1.0 if target_metric in LOWER_IS_BETTER_TARGETS else 1.0
    rows: list[dict[str, Any]] = []
    for driver, row in pivot.iterrows():
        low = _safe_float(row["Low"], 0.0)
        high = _safe_float(row["High"], 0.0)
        low_score = sign * low
        high_score = sign * high
        value_gain = max(0.0, low_score, high_score)
        leakage = max(0.0, -min(low_score, high_score))
        preferred_case = "High" if high_score >= low_score else "Low"
        preferred_delta = high if preferred_case == "High" else low
        rows.append(
            {
                "Driver": str(driver),
                "Value Gain Potential": value_gain,
                "Leakage Risk": leakage,
                "Preferred Case": preferred_case,
                "Preferred Delta": preferred_delta,
                "Sensitivity Abs": max(abs(low), abs(high)),
            }
        )
    return pd.DataFrame(rows).sort_values("Sensitivity Abs", ascending=False).reset_index(drop=True)


def _build_custom_chart_figure(df: pd.DataFrame, cfg: dict, slot: int) -> go.Figure | None:
    if df.empty:
        return None
    chart_type = str(cfg.get("chart_type", "line")).strip().lower()
    selected = cfg.get("columns", [])
    if not isinstance(selected, list):
        selected = []
    cols = [c for c in selected if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return None
    title = str(cfg.get("title", "")).strip() or f"Custom Chart {slot}"
    if chart_type == "area":
        melt = df[["Date"] + cols].melt("Date", var_name="Series", value_name="Value")
        return px.area(melt, x="Date", y="Value", color="Series", title=title)
    if chart_type == "bar":
        melt = df[["Date"] + cols].melt("Date", var_name="Series", value_name="Value")
        return px.bar(melt, x="Date", y="Value", color="Series", barmode="group", title=title)
    return px.line(df, x="Date", y=cols, title=title)


def build_pdf_chart_images(report_input: dict, options: dict | None = None) -> list[dict]:
    """Build chart images (or placeholders) for analyst PDF report sections."""

    options = _merge_options(options)
    width_px = _safe_int(options.get("chart_width_px", 1400), 1400)
    height_px = _safe_int(options.get("chart_height_px", 800), 800)

    range_df = _ensure_dataframe(report_input.get("range_df"))
    view_df = _ensure_dataframe(report_input.get("view_df"))
    assumptions = deepcopy(report_input.get("assumptions", {})) if isinstance(report_input, dict) else {}
    custom_df = _ensure_dataframe(report_input.get("custom_chart_data_df"))
    custom_cfgs = report_input.get("custom_chart_configs", []) if isinstance(report_input, dict) else []
    sensitivity_df = _ensure_dataframe(report_input.get("sensitivity_df"))
    sensitivity_target = str(report_input.get("sensitivity_target") or "Year N EBITDA")

    if range_df.empty and not view_df.empty:
        range_df = view_df.copy()
    if custom_df.empty:
        custom_df = range_df.copy()

    chart_images: list[dict[str, Any]] = []
    kaleido_ok = _ensure_kaleido_ready(options)
    if not kaleido_ok:
        _log_event(
            options,
            level="WARNING",
            event="pdf_chart_engine_unavailable",
            message="Kaleido is unavailable; PDF will include chart placeholders.",
            context={},
        )

    def _capture_chart(
        chart_id: str,
        title: str,
        section: str,
        fig_builder: Callable[[], go.Figure | None],
        *,
        core: bool = True,
    ) -> None:
        try:
            fig = fig_builder()
            if fig is None:
                chart_images.append(
                    _chart_placeholder(chart_id, title, section, "No data available for this chart.", core=core)
                )
                return
            if not kaleido_ok:
                chart_images.append(
                    _chart_placeholder(
                        chart_id, title, section, "Chart engine unavailable in this environment.", core=core
                    )
                )
                return
            chart_images.append(
                {
                    "id": chart_id,
                    "title": title,
                    "section": section,
                    "image_bytes": render_plotly_figure_png(fig, width_px=width_px, height_px=height_px),
                    "placeholder_text": "",
                    "core_chart": bool(core),
                }
            )
        except Exception as exc:
            _log_event(
                options,
                level="ERROR",
                event="pdf_chart_render_failed",
                message=f"Chart render failed for {chart_id}.",
                context={"chart_id": chart_id, "title": title},
                exc=exc,
            )
            chart_images.append(_chart_placeholder(chart_id, title, section, f"Chart render failed: {exc}", core=core))

    def _fig_revenue_by_segment() -> go.Figure | None:
        cols = [
            c
            for c in [
                "Service Revenue",
                "Replacement Revenue",
                "Maintenance Revenue",
                "Upsell Revenue",
                "New Build Revenue",
            ]
            if c in range_df.columns
        ]
        if not cols or "Date" not in range_df.columns:
            return None
        seg = range_df[["Date"] + cols].melt("Date", var_name="Segment", value_name="Revenue")
        return px.area(seg, x="Date", y="Revenue", color="Segment", title="Revenue by Segment")

    def _fig_ebitda_margin() -> go.Figure | None:
        if "Date" not in range_df.columns or "EBITDA" not in range_df.columns:
            return None
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=range_df["Date"], y=range_df["EBITDA"], name="EBITDA"))
        if "Total Revenue" in range_df.columns:
            margin = 100.0 * range_df["EBITDA"] / range_df["Total Revenue"].replace(0, pd.NA)
            fig.add_trace(go.Scatter(x=range_df["Date"], y=margin, name="EBITDA Margin %", yaxis="y2"))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Margin %"))
        fig.update_layout(title="EBITDA and EBITDA Margin")
        return fig

    def _fig_ending_cash() -> go.Figure | None:
        if "Date" not in range_df.columns or "End Cash" not in range_df.columns:
            return None
        return px.line(range_df, x="Date", y="End Cash", title="Ending Cash Balance")

    def _fig_margin_vs_opex() -> go.Figure | None:
        if "Date" not in range_df.columns or "Total Revenue" not in range_df.columns:
            return None
        if "Gross Profit" not in range_df.columns or "Total OPEX" not in range_df.columns:
            return None
        gm = 100.0 * range_df["Gross Profit"] / range_df["Total Revenue"].replace(0, pd.NA)
        op = 100.0 * range_df["Total OPEX"] / range_df["Total Revenue"].replace(0, pd.NA)
        frame = pd.DataFrame({"Date": range_df["Date"], "Gross Margin %": gm, "OPEX % Revenue": op})
        melt = frame.melt("Date", var_name="Metric", value_name="Percent")
        return px.line(melt, x="Date", y="Percent", color="Metric", title="Gross Margin % vs OPEX % Revenue")

    def _fig_fcf() -> go.Figure | None:
        if "Date" not in range_df.columns or "Free Cash Flow" not in range_df.columns:
            return None
        return px.line(range_df, x="Date", y="Free Cash Flow", title="Free Cash Flow Trend")

    def _fig_disbursements() -> go.Figure | None:
        if "Date" not in range_df.columns or "Total Disbursements" not in range_df.columns:
            return None
        return px.line(range_df, x="Date", y="Total Disbursements", title="Total Disbursements Trend")

    def _fig_cash_bridge() -> go.Figure | None:
        if range_df.empty:
            return None
        row = range_df.iloc[-1]
        needed = [
            "Total Revenue",
            "Total Direct Costs",
            "Total OPEX",
            "Change in NWC",
            "Capex",
            "Term Loan Payment",
            "LOC Interest",
            "LOC Draw",
            "LOC Repay",
            "Owner Distributions",
        ]
        for col in needed:
            if col not in range_df.columns:
                return None
        month_label = str(row.get("Year_Month_Label", "Selected End Month"))
        fig = go.Figure(
            go.Waterfall(
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
                connector={"line": {"color": "rgba(140,140,140,0.55)"}},
            )
        )
        fig.update_layout(title=f"Monthly Cash Flow Bridge ({month_label})", showlegend=False)
        return fig

    def _fig_ocf_capex_netcf() -> go.Figure | None:
        if "Date" not in range_df.columns:
            return None
        required = ["Operating Cash Flow", "Capex", "Net Cash Flow"]
        if any(c not in range_df.columns for c in required):
            return None
        fig = go.Figure()
        _add_series_if_present(fig, range_df, y_col="Operating Cash Flow", name="Operating Cash Flow")
        _add_series_if_present(fig, range_df, y_col="Capex", name="Capex")
        _add_series_if_present(fig, range_df, y_col="Net Cash Flow", name="Net Cash Flow")
        fig.update_layout(title="Operating Cash Flow vs Capex vs Net Cash Flow")
        return fig

    def _fig_debt_service() -> go.Figure | None:
        if "Date" not in range_df.columns:
            return None
        if "Term Loan Payment" not in range_df.columns and "LOC Interest" not in range_df.columns:
            return None
        fig = go.Figure()
        _add_series_if_present(fig, range_df, y_col="Term Loan Payment", name="Term Loan Payment")
        _add_series_if_present(fig, range_df, y_col="LOC Interest", name="LOC Interest")
        if "Term Loan Payment" in range_df.columns and "LOC Interest" in range_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=range_df["Date"],
                    y=range_df["Term Loan Payment"] + range_df["LOC Interest"],
                    name="Total Debt Service",
                    line=dict(width=4),
                )
            )
        fig.update_layout(title="Debt Service Trend")
        return fig

    def _fig_loan_loc_balance() -> go.Figure | None:
        if "Date" not in range_df.columns:
            return None
        if "Term Loan Balance" not in range_df.columns and "LOC Balance" not in range_df.columns:
            return None
        fig = go.Figure()
        _add_series_if_present(fig, range_df, y_col="Term Loan Balance", name="Term Loan Balance")
        _add_series_if_present(fig, range_df, y_col="LOC Balance", name="LOC Balance")
        fig.update_layout(title="Loan Balance and LOC Balance")
        return fig

    def _fig_staffing_fleet() -> go.Figure | None:
        if "Date" not in range_df.columns:
            return None
        base_cols = [c for c in ["Techs", "Sales Staff", "Trucks", "Retained Trucks"] if c in range_df.columns]
        if not base_cols:
            return None
        frame = range_df[["Date"] + base_cols].copy()
        if "Trucks" in frame.columns and "Retained Trucks" in frame.columns:
            frame["Total Fleet Units"] = frame["Trucks"] + frame["Retained Trucks"]
        melt = frame.melt("Date", var_name="Series", value_name="Value")
        return px.line(melt, x="Date", y="Value", color="Series", title="Staffing and Fleet Drivers")

    def _fig_calls_repl() -> go.Figure | None:
        if "Date" not in range_df.columns:
            return None
        cols = [c for c in ["Calls", "Replacement Leads", "Replacement Jobs"] if c in range_df.columns]
        if not cols:
            return None
        return px.line(range_df, x="Date", y=cols, title="Calls, Replacement Leads, and Replacement Jobs")

    def _fig_hours() -> go.Figure | None:
        if "Date" not in range_df.columns:
            return None
        cols = [c for c in ["Tech Hours", "Sales Hours"] if c in range_df.columns]
        if not cols:
            return None
        return px.line(range_df, x="Date", y=cols, title="Tech Hours vs Sales Hours")

    def _fig_fleet_cost_components() -> go.Figure | None:
        fc = _build_fleet_cost_components(range_df, assumptions)
        cols = [c for c in fc.columns if c != "Date"]
        if fc.empty or not cols:
            return None
        melt = fc.melt("Date", var_name="Component", value_name="Cost")
        return px.area(melt, x="Date", y="Cost", color="Component", title="Fleet Cost Components")

    def _fig_revenue_drivers() -> go.Figure | None:
        if "Date" not in range_df.columns:
            return None
        cols = [
            c
            for c in [
                "Service Revenue",
                "Replacement Revenue",
                "Maintenance Revenue",
                "Upsell Revenue",
                "New Build Revenue",
            ]
            if c in range_df.columns
        ]
        if not cols:
            return None
        return px.line(range_df, x="Date", y=cols, title="Revenue Drivers")

    def _fig_marketing_customers_cac() -> go.Figure | None:
        if "Date" not in range_df.columns or "Marketing Spend" not in range_df.columns:
            return None
        attach_rate = _safe_float(assumptions.get("attach_rate", 0.35), 0.35)
        calls = range_df["Calls"] if "Calls" in range_df.columns else 0.0
        repl_jobs = range_df["Replacement Jobs"] if "Replacement Jobs" in range_df.columns else 0.0
        new_customers = calls * attach_rate + repl_jobs
        monthly_cac = range_df["Marketing Spend"] / new_customers.replace(0, pd.NA)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=range_df["Date"], y=range_df["Marketing Spend"], name="Marketing Spend"))
        fig.add_trace(go.Scatter(x=range_df["Date"], y=new_customers, name="New Customers Proxy", yaxis="y2"))
        fig.add_trace(
            go.Scatter(
                x=range_df["Date"],
                y=monthly_cac,
                name="Monthly CAC",
                yaxis="y2",
                line=dict(dash="dot"),
            )
        )
        fig.update_layout(
            title="Marketing Spend vs New Customers Proxy vs CAC",
            yaxis=dict(title="Spend"),
            yaxis2=dict(title="Volume / CAC", overlaying="y", side="right"),
            barmode="group",
        )
        return fig

    def _fig_maintenance_by_segment() -> go.Figure | None:
        if "Date" not in range_df.columns:
            return None
        cols = [c for c in ["Res Maintenance Agreements", "LC Maintenance Agreements"] if c in range_df.columns]
        if not cols:
            return None
        return px.line(range_df, x="Date", y=cols, title="Maintenance Agreements by Segment")

    def _fig_new_build_segment() -> go.Figure | None:
        if "Date" not in range_df.columns:
            return None
        cols = [c for c in ["Res New Build Revenue", "LC New Build Revenue"] if c in range_df.columns]
        if not cols:
            return None
        return px.line(range_df, x="Date", y=cols, title="New-Build Revenue by Segment")

    def _fig_working_capital() -> go.Figure | None:
        if "Date" not in range_df.columns:
            return None
        cols = [c for c in ["AR Balance", "AP Balance", "Inventory Balance", "NWC"] if c in range_df.columns]
        if not cols:
            return None
        return px.line(range_df, x="Date", y=cols, title="AR/AP/Inventory Balances and NWC")

    def _fig_ocf_vs_nwc() -> go.Figure | None:
        if "Date" not in range_df.columns:
            return None
        required = ["Operating Cash Flow", "Change in NWC"]
        if any(c not in range_df.columns for c in required):
            return None
        fig = go.Figure()
        fig.add_trace(go.Bar(x=range_df["Date"], y=range_df["Change in NWC"], name="Change in NWC"))
        fig.add_trace(go.Scatter(x=range_df["Date"], y=range_df["Operating Cash Flow"], name="Operating Cash Flow"))
        fig.update_layout(title="Operating Cash Flow vs Change in NWC")
        return fig

    def _fig_tornado() -> go.Figure | None:
        if sensitivity_df.empty:
            return None
        delta_col = f"Delta {sensitivity_target}"
        if delta_col not in sensitivity_df.columns:
            return None
        tornado = sensitivity_df.pivot(index="Driver", columns="Case", values=delta_col).fillna(0.0)
        if "Low" not in tornado.columns:
            tornado["Low"] = 0.0
        if "High" not in tornado.columns:
            tornado["High"] = 0.0
        tornado["Base"] = 0.0
        melted = tornado[["Low", "Base", "High"]].reset_index().melt(
            id_vars="Driver", var_name="Case", value_name="Delta"
        )
        return px.bar(
            melted,
            x="Delta",
            y="Driver",
            color="Case",
            orientation="h",
            title=f"Tornado Chart for {sensitivity_target}",
        )

    def _fig_value_leakage_summary() -> go.Figure | None:
        summary = _build_sensitivity_summary_df(sensitivity_df, sensitivity_target)
        if summary.empty:
            return None
        top = summary.head(10).copy()
        top["Leakage Risk (negative)"] = -top["Leakage Risk"]
        chart = top[["Driver", "Value Gain Potential", "Leakage Risk (negative)"]].melt(
            id_vars="Driver", var_name="Series", value_name="Amount"
        )
        return px.bar(
            chart,
            x="Amount",
            y="Driver",
            color="Series",
            orientation="h",
            title="Top Value Leakage and Value Creation Drivers",
            barmode="relative",
        )

    core_specs = [
        ("core_01_revenue_segment", "Revenue by Segment", "Executive Trend Charts", _fig_revenue_by_segment),
        ("core_02_ebitda_margin", "EBITDA and EBITDA Margin", "Executive Trend Charts", _fig_ebitda_margin),
        ("core_03_ending_cash", "Ending Cash Balance", "Executive Trend Charts", _fig_ending_cash),
        ("core_04_margin_opex", "Gross Margin % vs OPEX % Revenue", "Executive Trend Charts", _fig_margin_vs_opex),
        ("core_05_fcf", "Free Cash Flow Trend", "Executive Trend Charts", _fig_fcf),
        ("core_06_disbursements", "Total Disbursements Trend", "Executive Trend Charts", _fig_disbursements),
        ("core_07_cash_bridge", "Monthly Cash Flow Bridge", "Cashflow & Financing Perspectives", _fig_cash_bridge),
        (
            "core_08_ocf_capex_netcf",
            "Operating Cash Flow vs Capex vs Net Cash Flow",
            "Cashflow & Financing Perspectives",
            _fig_ocf_capex_netcf,
        ),
        ("core_09_debt_service", "Debt Service Trend", "Cashflow & Financing Perspectives", _fig_debt_service),
        ("core_10_loan_loc", "Loan Balance and LOC Balance", "Cashflow & Financing Perspectives", _fig_loan_loc_balance),
        ("core_11_staffing_fleet", "Staffing and Fleet Drivers", "Operating Capacity & Fleet", _fig_staffing_fleet),
        (
            "core_12_calls_repl",
            "Calls, Replacement Leads, and Replacement Jobs",
            "Operating Capacity & Fleet",
            _fig_calls_repl,
        ),
        ("core_13_hours", "Tech Hours vs Sales Hours", "Operating Capacity & Fleet", _fig_hours),
        ("core_14_fleet_cost", "Fleet Cost Components", "Operating Capacity & Fleet", _fig_fleet_cost_components),
        ("core_15_revenue_drivers", "Revenue Drivers", "Commercial Mix & Growth Drivers", _fig_revenue_drivers),
        (
            "core_16_marketing_cac",
            "Marketing Spend vs New Customers Proxy vs CAC",
            "Commercial Mix & Growth Drivers",
            _fig_marketing_customers_cac,
        ),
        (
            "core_17_maint_segment",
            "Maintenance Agreements by Segment",
            "Commercial Mix & Growth Drivers",
            _fig_maintenance_by_segment,
        ),
        (
            "core_18_new_build_segment",
            "New-Build Revenue by Segment",
            "Commercial Mix & Growth Drivers",
            _fig_new_build_segment,
        ),
        (
            "core_19_wc_balances",
            "AR/AP/Inventory Balances and NWC",
            "Working Capital & Liquidity",
            _fig_working_capital,
        ),
        (
            "core_20_ocf_nwc",
            "Operating Cash Flow vs Change in NWC",
            "Working Capital & Liquidity",
            _fig_ocf_vs_nwc,
        ),
        ("core_21_tornado", "Tornado Chart", "Risk & Sensitivity", _fig_tornado),
        ("core_22_leakage", "Value Leakage / Creation Summary", "Risk & Sensitivity", _fig_value_leakage_summary),
    ]
    for chart_id, title, section, fig_builder in core_specs:
        _capture_chart(chart_id, title, section, fig_builder, core=True)

    include_custom = bool(options.get("include_enabled_custom_charts", True))
    if include_custom and isinstance(custom_cfgs, list):
        for slot, cfg in enumerate(custom_cfgs[:4], start=1):
            if not isinstance(cfg, dict):
                continue
            if not bool(cfg.get("enabled", False)):
                continue
            title = str(cfg.get("title") or f"Custom Chart {slot}")
            _capture_chart(
                f"custom_chart_{slot}",
                title,
                "Enabled Custom Charts",
                lambda cfg=cfg, slot=slot: _build_custom_chart_figure(custom_df, cfg, slot),
                core=False,
            )

    return chart_images


def _build_kpi_snapshot_table(metrics_range: dict, metrics_full: dict) -> pd.DataFrame:
    def _metric_total(metrics: dict, frame_key: str, column: str, scalar_key: str) -> float:
        value = metrics.get(frame_key)
        if isinstance(value, pd.DataFrame) and column in value.columns:
            return _safe_float(value[column].sum())
        return _safe_float(metrics.get(scalar_key, 0.0))

    rows = [
        (
            "Total Revenue",
            _fmt_currency(_metric_total(metrics_range, "revenue_by_year", "Total Revenue", "total_revenue")),
            _fmt_currency(_metric_total(metrics_full, "revenue_by_year", "Total Revenue", "total_revenue")),
        ),
        (
            "Total EBITDA",
            _fmt_currency(_metric_total(metrics_range, "ebitda_by_year", "EBITDA", "total_ebitda")),
            _fmt_currency(_metric_total(metrics_full, "ebitda_by_year", "EBITDA", "total_ebitda")),
        ),
        (
            "Total Free Cash Flow",
            _fmt_currency(_metric_total(metrics_range, "fcf_by_year", "Free Cash Flow", "total_fcf")),
            _fmt_currency(_metric_total(metrics_full, "fcf_by_year", "Free Cash Flow", "total_fcf")),
        ),
        ("Minimum Ending Cash", _fmt_currency(metrics_range.get("minimum_ending_cash", 0.0)), _fmt_currency(metrics_full.get("minimum_ending_cash", 0.0))),
        ("Negative Cash Months", str(_safe_int(metrics_range.get("negative_cash_months", 0))), str(_safe_int(metrics_full.get("negative_cash_months", 0)))),
        ("Avg Gross Margin", _fmt_percent(metrics_range.get("gross_margin_full_period_avg", 0.0)), _fmt_percent(metrics_full.get("gross_margin_full_period_avg", 0.0))),
        ("CAC", _fmt_currency_2(metrics_range.get("cac", 0.0)), _fmt_currency_2(metrics_full.get("cac", 0.0))),
        ("Cash Conversion Cycle", f"{_fmt_number(metrics_range.get('ccc', 0.0))} days", f"{_fmt_number(metrics_full.get('ccc', 0.0))} days"),
        ("Break-even Revenue", _fmt_currency(metrics_range.get("break_even_revenue", 0.0)), _fmt_currency(metrics_full.get("break_even_revenue", 0.0))),
        ("Break-even Labor Rate", f"{_fmt_currency_2(metrics_range.get('break_even_labor_rate_per_tech_hour', 0.0))}/hr", f"{_fmt_currency_2(metrics_full.get('break_even_labor_rate_per_tech_hour', 0.0))}/hr"),
        ("Break-even Wage Rate", f"{_fmt_currency_2(metrics_range.get('break_even_wage_rate_per_hour', 0.0))}/hr", f"{_fmt_currency_2(metrics_full.get('break_even_wage_rate_per_hour', 0.0))}/hr"),
        ("Total Disbursements", _fmt_currency(metrics_range.get("total_disbursements", 0.0)), _fmt_currency(metrics_full.get("total_disbursements", 0.0))),
    ]
    return pd.DataFrame(rows, columns=["KPI", "Selected Range", "Full Horizon Reference"])


def _build_assumptions_catalog_table(assumptions: dict) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key in sorted(assumptions.keys()):
        value = assumptions.get(key)
        guidance = INPUT_GUIDANCE.get(key, {})
        rows.append(
            {
                "Input": key,
                "Value": _fmt_value(value),
                "Guidance Min": _fmt_value(guidance.get("min")),
                "Guidance Max": _fmt_value(guidance.get("max")),
                "Guidance Note": str(guidance.get("note", "")),
            }
        )
    return pd.DataFrame(rows)


def _build_monthly_domain_tables(view_df: pd.DataFrame) -> list[dict[str, Any]]:
    if view_df.empty:
        return []
    prefix = ["Year_Month_Label"] if "Year_Month_Label" in view_df.columns else ["Date"] if "Date" in view_df.columns else []
    groups = [
        (
            "Monthly Detail - Operations and Demand",
            [
                "Workdays",
                "Techs",
                "Sales Staff",
                "Tech Hours",
                "Sales Hours",
                "Calls",
                "Replacement Leads",
                "Replacement Jobs",
                "Res Maintenance Agreements",
                "LC Maintenance Agreements",
            ],
        ),
        (
            "Monthly Detail - Revenue Composition",
            [
                "Service Revenue",
                "Replacement Revenue",
                "Maintenance Revenue",
                "Upsell Revenue",
                "New Build Revenue",
                "Total Revenue",
                "Res New Build Revenue",
                "LC New Build Revenue",
            ],
        ),
        (
            "Monthly Detail - Direct Costs",
            [
                "Service Materials",
                "Replacement Equipment",
                "Permits",
                "Disposal",
                "Direct Labor",
                "Maintenance Direct Cost",
                "Upsell Direct Cost",
                "New Build Direct Cost",
                "Total Direct Costs",
                "Gross Profit",
            ],
        ),
        (
            "Monthly Detail - OPEX and Payroll",
            [
                "Fixed OPEX",
                "Marketing Spend",
                "Fleet Cost",
                "Sales Payroll",
                "Management Payroll",
                "Total OPEX",
                "EBITDA",
            ],
        ),
        (
            "Monthly Detail - Working Capital",
            ["AR Balance", "AP Balance", "Inventory Balance", "NWC", "Change in NWC", "Operating Cash Flow"],
        ),
        (
            "Monthly Detail - Financing and Cash",
            [
                "Capex",
                "Free Cash Flow",
                "Term Loan Payment",
                "LOC Interest",
                "LOC Draw",
                "LOC Repay",
                "LOC Balance",
                "Owner Distributions",
                "Net Cash Flow",
                "End Cash",
                "Total Disbursements",
            ],
        ),
    ]
    out: list[dict[str, Any]] = []
    for title, cols in groups:
        use_cols = prefix + [c for c in cols if c in view_df.columns]
        if len(use_cols) <= len(prefix):
            continue
        out.append({"title": title, "dataframe": view_df[use_cols].copy()})
    return out


def _input_month_column(df: pd.DataFrame) -> str:
    if "Year_Month_Label" in df.columns:
        return "Year_Month_Label"
    if "Date" in df.columns:
        return "Date"
    return ""


def _build_input_timeseries_overview_table(input_ts_df: pd.DataFrame) -> pd.DataFrame:
    if input_ts_df.empty:
        return pd.DataFrame()
    month_col = _input_month_column(input_ts_df)
    cols = [c for c in input_ts_df.columns if c != month_col and c != "Date"]
    if not cols:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for col in cols:
        series = input_ts_df[col]
        start_val = series.iloc[0] if len(series) else None
        end_val = series.iloc[-1] if len(series) else None
        changed = bool(series.nunique(dropna=False) > 1)
        non_zero_months = ""
        if pd.api.types.is_numeric_dtype(series):
            non_zero_months = int((pd.to_numeric(series, errors="coerce").fillna(0.0).abs() > 1e-9).sum())
        rows.append(
            {
                "Input": col,
                "Start Value": _fmt_value(start_val),
                "End Value": _fmt_value(end_val),
                "Changed Over Time": "Yes" if changed else "No",
                "Non-Zero Months": non_zero_months,
            }
        )
    return pd.DataFrame(rows).sort_values("Input").reset_index(drop=True)


def _build_event_matrix_table(input_ts_df: pd.DataFrame) -> pd.DataFrame:
    if input_ts_df.empty:
        return pd.DataFrame()
    month_col = _input_month_column(input_ts_df)
    if not month_col:
        return pd.DataFrame()
    event_name_map = {
        "tech_staffing_events_hires_input": "Tech Hires",
        "tech_staffing_events_attrition_input": "Tech Attrition",
        "sales_staffing_events_hires_input": "Sales Hires",
        "sales_staffing_events_attrition_input": "Sales Attrition",
        "res_new_build_install_schedule_installs_input": "Residential New-Build Installs",
        "lc_new_build_install_schedule_installs_input": "LC New-Build Installs",
    }
    event_cols = [c for c in event_name_map.keys() if c in input_ts_df.columns]
    if not event_cols:
        return pd.DataFrame()

    frame = pd.DataFrame({"Month": input_ts_df[month_col]})
    for col in event_cols:
        frame[event_name_map[col]] = pd.to_numeric(input_ts_df[col], errors="coerce").fillna(0.0)

    metric_cols = [c for c in frame.columns if c != "Month"]
    active_mask = frame[metric_cols].abs().sum(axis=1) > 1e-9
    active = frame.loc[active_mask].reset_index(drop=True)
    if active.empty:
        return pd.DataFrame(
            {
                "Month": ["(none)"],
                "Note": ["No non-zero staffing or install schedule events in the selected horizon."],
            }
        )
    return active


def _build_input_series_change_log_table(input_ts_df: pd.DataFrame) -> pd.DataFrame:
    if input_ts_df.empty:
        return pd.DataFrame()
    month_col = _input_month_column(input_ts_df)
    if not month_col:
        return pd.DataFrame()

    skip_cols = {
        month_col,
        "Date",
        "tech_staffing_events_hires_input",
        "tech_staffing_events_attrition_input",
        "sales_staffing_events_hires_input",
        "sales_staffing_events_attrition_input",
        "res_new_build_install_schedule_installs_input",
        "lc_new_build_install_schedule_installs_input",
    }

    rows: list[dict[str, Any]] = []
    for col in input_ts_df.columns:
        if col in skip_cols:
            continue
        series = input_ts_df[col]
        if series.nunique(dropna=False) <= 1:
            continue
        prev = series.shift(1)
        changed_mask = series.astype(str) != prev.astype(str)
        changed_idx = list(input_ts_df.index[changed_mask])
        for idx in changed_idx:
            rows.append(
                {
                    "Month": _fmt_value(input_ts_df.loc[idx, month_col]),
                    "Input": col,
                    "Value": _fmt_value(input_ts_df.loc[idx, col]),
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["Month", "Input"]).reset_index(drop=True)


def _build_input_timeseries_appendix_tables(input_ts_df: pd.DataFrame) -> list[dict[str, Any]]:
    if input_ts_df.empty:
        return []
    tables: list[dict[str, Any]] = []
    overview = _build_input_timeseries_overview_table(input_ts_df)
    if not overview.empty:
        tables.append({"title": "Input Time-Series Overview (Start vs End)", "dataframe": overview})

    event_matrix = _build_event_matrix_table(input_ts_df)
    if not event_matrix.empty:
        tables.append({"title": "Staffing and Install Event Schedule (Non-zero Months)", "dataframe": event_matrix})

    change_log = _build_input_series_change_log_table(input_ts_df)
    if not change_log.empty:
        tables.append({"title": "Input Series Change Log (Only Changed Months)", "dataframe": change_log})
    return tables


def build_report_sections(report_input: dict, chart_images: list[dict], options: dict | None = None) -> list[dict]:
    """Build report section descriptors consumed by PDF rendering."""

    options = _merge_options(options)
    scenario_name = str(report_input.get("scenario_name") or "Current Scenario")
    generated_at = str(report_input.get("generated_at_utc") or _utc_iso_now())
    value_mode = str(report_input.get("value_mode") or "nominal")
    range_start = str(report_input.get("range_start_label") or "")
    range_end = str(report_input.get("range_end_label") or "")
    model_version = str(report_input.get("model_version") or "v2")
    metrics_range = report_input.get("metrics_range", {}) if isinstance(report_input, dict) else {}
    metrics_full = report_input.get("metrics_full", {}) if isinstance(report_input, dict) else {}
    annual_kpis_range = _ensure_dataframe(report_input.get("annual_kpis_range"))
    annual_kpis_full = _ensure_dataframe(report_input.get("annual_kpis_full"))
    assumptions = deepcopy(report_input.get("assumptions", {})) if isinstance(report_input, dict) else {}
    input_ts_df = _ensure_dataframe(report_input.get("input_ts_df"))
    view_df = _ensure_dataframe(report_input.get("view_df"))
    transformation_logic = deepcopy(report_input.get("transformation_logic", {})) if isinstance(report_input, dict) else {}
    warnings = report_input.get("input_warnings", []) if isinstance(report_input, dict) else []
    integrity_findings = report_input.get("integrity_findings", []) if isinstance(report_input, dict) else []

    chart_map: dict[str, list[dict]] = defaultdict(list)
    for item in chart_images:
        section = str(item.get("section", "Other"))
        chart_map[section].append(item)

    sections: list[dict[str, Any]] = [
        {
            "id": "cover",
            "title": "Cover + Report Context",
            "paragraphs": [
                f"Scenario: {scenario_name}",
                f"Generated (UTC): {generated_at}",
                f"Selected Range: {range_start} to {range_end}",
                f"Value Mode: {value_mode}",
                f"Model Version: {model_version}",
                "How to read this report: KPI values are shown for the selected range with full-horizon references where relevant.",
                "Charts primarily use selected-range data while appendix tables include full-horizon context.",
            ],
            "tables": [],
            "charts": [],
        },
        {
            "id": "kpi_snapshot",
            "title": "Executive KPI Snapshot",
            "paragraphs": [],
            "tables": [{"title": "Headline KPI Table", "dataframe": _build_kpi_snapshot_table(metrics_range, metrics_full)}],
            "charts": [],
        },
        {
            "id": "executive_trends",
            "title": "Executive Trend Charts",
            "paragraphs": [],
            "tables": [],
            "charts": chart_map.get("Executive Trend Charts", []),
        },
        {
            "id": "cashflow_financing",
            "title": "Cashflow & Financing Perspectives",
            "paragraphs": [],
            "tables": [],
            "charts": chart_map.get("Cashflow & Financing Perspectives", []),
        },
        {
            "id": "ops_fleet",
            "title": "Operating Capacity & Fleet",
            "paragraphs": [],
            "tables": [],
            "charts": chart_map.get("Operating Capacity & Fleet", []),
        },
        {
            "id": "commercial_growth",
            "title": "Commercial Mix & Growth Drivers",
            "paragraphs": [],
            "tables": [],
            "charts": chart_map.get("Commercial Mix & Growth Drivers", []),
        },
        {
            "id": "working_capital",
            "title": "Working Capital & Liquidity",
            "paragraphs": [],
            "tables": [],
            "charts": chart_map.get("Working Capital & Liquidity", []),
        },
        {
            "id": "risk_sensitivity",
            "title": "Risk & Sensitivity",
            "paragraphs": [
                "Sensitivity deltas represent one-way shocks to selected drivers. For break-even and disbursement targets, lower outcomes are favorable."
            ],
            "tables": [],
            "charts": chart_map.get("Risk & Sensitivity", []),
        },
    ]

    custom_charts = chart_map.get("Enabled Custom Charts", [])
    if custom_charts:
        sections.append(
            {
                "id": "custom_charts",
                "title": "Enabled Custom Charts",
                "paragraphs": ["Includes all user-enabled custom charts (up to 4)."],
                "tables": [],
                "charts": custom_charts,
            }
        )

    appendix_tables: list[dict[str, Any]] = []
    appendix_tables.append({"title": "Assumptions Catalog with Guidance", "dataframe": _build_assumptions_catalog_table(assumptions)})
    if isinstance(warnings, list) and warnings:
        appendix_tables.append(
            {"title": "Input Warnings", "dataframe": pd.DataFrame({"Warning": [str(w) for w in warnings]})}
        )
    if isinstance(integrity_findings, list) and integrity_findings:
        appendix_tables.append(
            {"title": "Accounting Integrity Findings", "dataframe": pd.DataFrame(integrity_findings)}
        )
    if not annual_kpis_range.empty:
        appendix_tables.append({"title": "Annual KPIs - Selected Range", "dataframe": annual_kpis_range})
    if not annual_kpis_full.empty:
        appendix_tables.append({"title": "Annual KPIs - Full Horizon", "dataframe": annual_kpis_full})
    appendix_tables.extend(_build_monthly_domain_tables(view_df))
    appendix_tables.extend(_build_input_timeseries_appendix_tables(input_ts_df))

    if isinstance(transformation_logic, dict):
        pipeline_steps = transformation_logic.get("pipeline_steps", [])
        if isinstance(pipeline_steps, list) and pipeline_steps:
            appendix_tables.append(
                {
                    "title": "Transformation Pipeline Steps",
                    "dataframe": pd.DataFrame({"Step": [str(s) for s in pipeline_steps]}),
                }
            )
        core_id = transformation_logic.get("core_identities", [])
        if isinstance(core_id, list) and core_id:
            appendix_tables.append({"title": "Core Formula Identities", "dataframe": pd.DataFrame(core_id)})
        value_logic = transformation_logic.get("value_mode_logic", [])
        if isinstance(value_logic, list) and value_logic:
            appendix_tables.append({"title": "Value Mode Logic", "dataframe": pd.DataFrame(value_logic)})
        refs = transformation_logic.get("implementation_references", [])
        if isinstance(refs, list) and refs:
            appendix_tables.append({"title": "Implementation References", "dataframe": pd.DataFrame(refs)})

    sections.append(
        {
            "id": "appendix",
            "title": "Maximum Detail Appendix",
            "paragraphs": ["Detailed assumptions, annual KPIs, monthly tables, input time-series, and transformation logic."],
            "tables": appendix_tables,
            "charts": [],
        }
    )

    return sections


def _reportlab_imports():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    return {
        "colors": colors,
        "letter": letter,
        "landscape": landscape,
        "ParagraphStyle": ParagraphStyle,
        "getSampleStyleSheet": getSampleStyleSheet,
        "inch": inch,
        "Image": Image,
        "PageBreak": PageBreak,
        "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
        "Table": Table,
        "TableStyle": TableStyle,
    }


def _printable_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{float(x):,.4f}".rstrip("0").rstrip("."))
        else:
            out[col] = out[col].map(_fmt_value)
        out[col] = out[col].astype(str).map(lambda s: s[:260])
    return out


def _append_dataframe_table_to_story(story: list[Any], table_spec: dict, rl: dict, styles: dict) -> None:
    Paragraph = rl["Paragraph"]
    Spacer = rl["Spacer"]
    Table = rl["Table"]
    TableStyle = rl["TableStyle"]
    colors = rl["colors"]
    inch = rl["inch"]

    title = str(table_spec.get("title", "Table"))
    df = _ensure_dataframe(table_spec.get("dataframe"))
    story.append(Paragraph(title, styles["Heading3"]))
    if df.empty:
        story.append(Paragraph("No data available.", styles["BodyText"]))
        story.append(Spacer(1, 8))
        return

    def _safe_cell_text(value: Any, max_len: int = 320) -> str:
        text = str(value)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = " ".join(text.split())
        if len(text) > max_len:
            text = f"{text[: max_len - 3].rstrip()}..."
        return html.escape(text)

    def _calc_col_widths(frame_in: pd.DataFrame, total_width: float) -> list[float]:
        sample = frame_in.head(min(30, len(frame_in)))
        weights: list[float] = []
        for col in frame_in.columns:
            max_len = max(len(str(col)), 6)
            if len(sample):
                max_len = max(max_len, int(sample[col].astype(str).map(len).max()))
            weights.append(float(max(6, min(28, max_len))))
        total_weight = sum(weights) or 1.0
        raw = [total_width * (w / total_weight) for w in weights]
        min_width = 0.75 * inch
        adjusted = [max(min_width, w) for w in raw]
        adjusted_total = sum(adjusted)
        if adjusted_total > total_width and adjusted_total > 0:
            scale = total_width / adjusted_total
            adjusted = [w * scale for w in adjusted]
        return adjusted

    frame = _printable_dataframe(df)
    max_cols = 6
    max_rows = 28
    for col_start in range(0, len(frame.columns), max_cols):
        subset = frame.iloc[:, col_start : col_start + max_cols]
        if col_start > 0:
            story.append(Paragraph(f"Column continuation {col_start + 1}-{col_start + len(subset.columns)}", styles["Small"]))
        for row_start in range(0, len(subset), max_rows):
            chunk = subset.iloc[row_start : row_start + max_rows]
            header_row = [Paragraph(_safe_cell_text(c), styles["TableHeader"]) for c in chunk.columns]
            data_rows: list[list[Any]] = []
            for _, row in chunk.iterrows():
                data_rows.append([Paragraph(_safe_cell_text(v), styles["TableCell"]) for v in row.tolist()])
            rows = [header_row] + data_rows
            col_widths = _calc_col_widths(subset, total_width=10.2 * inch)
            t = Table(rows, repeatRows=1, colWidths=col_widths)
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e6f2")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#102a43")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 6.6),
                        ("LEADING", (0, 0), (-1, -1), 8.0),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bcccdc")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                    ]
                )
            )
            story.append(t)
            story.append(Spacer(1, 8))


def _build_reportlab_pdf(report_input: dict, sections: list[dict], options: dict) -> bytes:
    rl = _reportlab_imports()
    SimpleDocTemplate = rl["SimpleDocTemplate"]
    Paragraph = rl["Paragraph"]
    Spacer = rl["Spacer"]
    Image = rl["Image"]
    PageBreak = rl["PageBreak"]
    getSampleStyleSheet = rl["getSampleStyleSheet"]
    ParagraphStyle = rl["ParagraphStyle"]
    landscape = rl["landscape"]
    letter = rl["letter"]
    inch = rl["inch"]

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.0,
            leading=8.4,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=6.4,
            leading=7.8,
            wordWrap="CJK",
        )
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(letter),
        leftMargin=0.4 * inch,
        rightMargin=0.4 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
        title=str(report_input.get("scenario_name", "HVAC Analyst Report")),
    )

    story: list[Any] = []
    for section_idx, section in enumerate(sections):
        story.append(Paragraph(str(section.get("title", "Section")), styles["Heading1"]))
        for para in section.get("paragraphs", []):
            story.append(Paragraph(str(para), styles["BodyText"]))
        if section.get("paragraphs"):
            story.append(Spacer(1, 8))

        for chart in section.get("charts", []):
            story.append(Paragraph(str(chart.get("title", "Chart")), styles["Heading3"]))
            image_bytes = chart.get("image_bytes")
            if image_bytes:
                img = Image(BytesIO(image_bytes))
                img.drawWidth = 9.8 * inch
                img.drawHeight = 5.2 * inch
                story.append(img)
            else:
                placeholder = str(chart.get("placeholder_text", "Chart unavailable."))
                story.append(Paragraph(placeholder, styles["BodyText"]))
            story.append(Spacer(1, 10))

        for table_spec in section.get("tables", []):
            _append_dataframe_table_to_story(story, table_spec, rl, styles)

        if section_idx < len(sections) - 1:
            story.append(PageBreak())

    doc.build(story)
    return buf.getvalue()


def _build_fallback_text_pdf(report_input: dict, sections: list[dict]) -> bytes:
    lines = [
        "HVAC Analyst Report",
        f"Scenario: {report_input.get('scenario_name', 'Current Scenario')}",
        f"Generated (UTC): {report_input.get('generated_at_utc', _utc_iso_now())}",
        "",
    ]
    for section in sections:
        lines.append(str(section.get("title", "Section")))
        for para in section.get("paragraphs", []):
            lines.append(f" - {para}")
        for chart in section.get("charts", []):
            lines.append(f" - Chart: {chart.get('title', 'Untitled')} ({'ok' if chart.get('image_bytes') else 'placeholder'})")
        for table_spec in section.get("tables", []):
            df = _ensure_dataframe(table_spec.get("dataframe"))
            lines.append(f" - Table: {table_spec.get('title', 'Table')} [{len(df)} rows, {len(df.columns)} cols]")
        lines.append("")
    return _build_minimal_pdf(lines)


def build_analyst_pdf_report_bytes(report_input: dict, options: dict) -> bytes:
    """Build analyst-grade PDF bytes with resilient chart fallback behavior."""

    merged_options = _merge_options(options)
    chart_images = merged_options.get("chart_images_override")
    if not isinstance(chart_images, list):
        chart_images = build_pdf_chart_images(report_input, merged_options)
    sections = build_report_sections(report_input, chart_images, merged_options)
    reportlab_ready = _ensure_reportlab_ready(merged_options)
    try:
        if not reportlab_ready:
            raise RuntimeError("ReportLab unavailable.")
        pdf_bytes = _build_reportlab_pdf(report_input, sections, merged_options)
        if not pdf_bytes.startswith(b"%PDF"):
            raise RuntimeError("ReportLab returned unexpected output.")
        return pdf_bytes
    except Exception as exc:
        _log_event(
            merged_options,
            level="WARNING",
            event="pdf_export_reportlab_fallback",
            message="ReportLab unavailable or failed; using minimal PDF fallback.",
            context={},
            exc=exc,
        )
        return _build_fallback_text_pdf(report_input, sections)
