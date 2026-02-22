from __future__ import annotations

import json
import uuid

import pandas as pd
from streamlit.testing.v1 import AppTest


def _widget_by_label(widgets, label: str):
    matches = [w for w in widgets if getattr(w, "label", "") == label]
    assert matches, f"Widget not found for label: {label}"
    return matches[0]


def _assert_no_app_exceptions(at: AppTest) -> None:
    assert len(at.exception) == 0


def test_app_initial_run_has_no_exceptions():
    at = AppTest.from_file("app.py")
    at.run(timeout=180)
    _assert_no_app_exceptions(at)


def test_workspace_load_and_goal_seek_smoke_flow():
    at = AppTest.from_file("app.py")
    at.run(timeout=180)
    _assert_no_app_exceptions(at)

    save_name = f"smoke_ws_{uuid.uuid4().hex[:8]}"

    # Save a workspace in manual mode, then load it after switching back to live mode.
    at.toggle(key="auto_run_model").set_value(False)
    at.run(timeout=180)
    _widget_by_label(at.text_input, "Save Name").set_value(save_name)
    at.run(timeout=180)
    _widget_by_label(at.button, "Save Workspace").click()
    at.run(timeout=180)
    _assert_no_app_exceptions(at)

    at.toggle(key="auto_run_model").set_value(True)
    at.run(timeout=180)
    _widget_by_label(at.selectbox, "Saved Workspaces").set_value(save_name)
    at.run(timeout=180)
    _widget_by_label(at.button, "Load Workspace").click()
    at.run(timeout=180)
    _assert_no_app_exceptions(at)

    # Apply deferred widget-state updates from workspace load.
    at.run(timeout=180)
    assert at.toggle(key="auto_run_model").value is False

    # Exercise goal seek interaction.
    at.number_input(key="goal_target_value").set_value(2_000_000.0)
    at.run(timeout=180)
    _widget_by_label(at.button, "Run Goal Seek").click()
    at.run(timeout=180)
    _assert_no_app_exceptions(at)
    assert "goal_seek_result" in at.session_state
    assert at.session_state["goal_seek_result"] is not None

    # Exercise helper quick-shift flow, including deferred reset of slider widget keys.
    at.slider(key="helper_revenue_shift_pct").set_value(0.05)
    at.slider(key="helper_cost_shift_pct").set_value(0.02)
    at.run(timeout=180)
    _widget_by_label(at.button, "Apply Quick Shift").click()
    at.run(timeout=180)
    _assert_no_app_exceptions(at)
    at.run(timeout=180)
    assert at.slider(key="helper_revenue_shift_pct").value == 0.0
    assert at.slider(key="helper_cost_shift_pct").value == 0.0

    # Generate AI export with active + selected saved items.
    at.radio(key="ai_export_scope").set_value("Active + selected saved items")
    at.run(timeout=180)
    at.multiselect(key="ai_export_selected_workspaces").set_value([save_name])
    at.run(timeout=180)
    _widget_by_label(at.button, "Generate AI Export Pack").click()
    at.run(timeout=180)
    _assert_no_app_exceptions(at)
    assert "ai_export_payload_json" in at.session_state
    payload_json = at.session_state["ai_export_payload_json"]
    assert isinstance(payload_json, str) and len(payload_json) > 1000
    payload = json.loads(payload_json)
    assert payload["export_type"] == "hvac_ai_context_pack"
    assert payload["selection"]["scope"] == "Active + selected saved items"
    assert int(payload["summary"]["scenario_count"]) >= 2

    # Clean up saved workspace.
    _widget_by_label(at.selectbox, "Saved Workspaces").set_value(save_name)
    at.run(timeout=180)
    _widget_by_label(at.button, "Delete Workspace").click()
    at.run(timeout=180)
    _assert_no_app_exceptions(at)


def test_last_active_input_section_tracks_recent_edits():
    at = AppTest.from_file("app.py")
    at.run(timeout=180)
    assert at.session_state["last_active_input_section"] == "Model Controls"

    at.number_input(key="avg_service_ticket").set_value(400.0)
    at.run(timeout=180)
    assert at.session_state["last_active_input_section"] == "Service and Replacement"

    at.slider(key="res_maint_hybrid_weight_calls").set_value(0.55)
    at.run(timeout=180)
    assert at.session_state["last_active_input_section"] == "Maintenance"

    at.selectbox(key="capex_trucks_mode").set_value("payments_only")
    at.run(timeout=180)
    assert at.session_state["last_active_input_section"] == "Marketing and Fleet"


def test_interactive_widgets_expose_help_tooltips():
    at = AppTest.from_file("app.py")
    at.run(timeout=180)
    _assert_no_app_exceptions(at)

    widget_groups = {
        "number_input": at.number_input,
        "slider": at.slider,
        "selectbox": at.selectbox,
        "toggle": at.toggle,
        "checkbox": at.checkbox,
        "multiselect": at.multiselect,
        "text_input": at.text_input,
        "radio": at.radio,
        "button": at.button,
    }
    for widget_type, widgets in widget_groups.items():
        missing = [
            getattr(widget, "label", "<no label>")
            for widget in widgets
            if not isinstance(getattr(widget, "help", None), str) or not str(widget.help).strip()
        ]
        assert not missing, f"{widget_type} widgets missing help: {', '.join(missing[:5])}"


def test_sensitivity_target_missing_delta_column_auto_recovers():
    at = AppTest.from_file("app.py")
    at.run(timeout=180)
    _assert_no_app_exceptions(at)

    # Simulate stale sensitivity schema persisted from an older app build.
    stale_df = pd.DataFrame(
        [
            {"Driver": "tech_wage_per_hour", "Case": "Low", "Delta Year N EBITDA": 1250.0},
            {"Driver": "tech_wage_per_hour", "Case": "High", "Delta Year N EBITDA": -1100.0},
        ]
    )
    at.session_state["sensitivity_result_df"] = stale_df
    if not at.session_state["sensitivity_drivers"]:
        at.session_state["sensitivity_drivers"] = ["tech_wage_per_hour"]

    _widget_by_label(at.selectbox, "Target metric").set_value("Break-even Labor Rate")
    at.run(timeout=180)
    _assert_no_app_exceptions(at)
    recovered_df = at.session_state["sensitivity_result_df"] if "sensitivity_result_df" in at.session_state else None
    assert recovered_df is not None
    assert "Delta Break-even Labor Rate" in recovered_df.columns


def test_analyst_pdf_controls_generate_payload_and_download_button():
    at = AppTest.from_file("app.py")
    at.run(timeout=240)
    _assert_no_app_exceptions(at)

    _widget_by_label(at.button, "Generate Analyst PDF").click()
    at.run(timeout=240)
    _assert_no_app_exceptions(at)

    assert "pdf_export_payload_bytes" in at.session_state
    payload = at.session_state["pdf_export_payload_bytes"]
    assert isinstance(payload, (bytes, bytearray))
    assert len(payload) > 500
    assert "pdf_export_filename" in at.session_state
    assert str(at.session_state["pdf_export_filename"]).endswith(".pdf")


def test_analyst_pdf_generation_supports_custom_charts_and_engine_fallback():
    at = AppTest.from_file("app.py")
    at.run(timeout=240)
    _assert_no_app_exceptions(at)

    at.checkbox(key="chart_1_enabled").set_value(True)
    at.text_input(key="chart_1_title").set_value("Custom Smoke Chart")
    at.selectbox(key="chart_1_type").set_value("line")
    at.multiselect(key="chart_1_cols").set_value(["Total Revenue", "EBITDA"])
    at.run(timeout=240)

    _widget_by_label(at.button, "Generate Analyst PDF").click()
    at.run(timeout=240)
    _assert_no_app_exceptions(at)

    assert "pdf_export_summary" in at.session_state
    summary = at.session_state["pdf_export_summary"]
    assert isinstance(summary, dict)
    assert int(summary.get("chart_count", 0)) >= 23
    assert int(summary.get("chart_placeholders", 0)) >= 0


def test_storage_location_custom_path_apply_flow():
    at = AppTest.from_file("app.py")
    at.run(timeout=240)
    _assert_no_app_exceptions(at)

    custom_rel = f".local_store/storage_smoke_{uuid.uuid4().hex[:8]}"
    _widget_by_label(at.radio, "Storage Mode").set_value("Custom server folder path")
    at.run(timeout=240)
    _widget_by_label(at.text_input, "Custom Storage Folder").set_value(custom_rel)
    at.run(timeout=240)
    _widget_by_label(at.button, "Apply Storage Location").click()
    at.run(timeout=240)
    _assert_no_app_exceptions(at)

    assert "storage_active_path" in at.session_state
    assert str(at.session_state["storage_active_path"])
