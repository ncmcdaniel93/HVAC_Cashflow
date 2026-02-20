from __future__ import annotations

from copy import deepcopy

from src.integrity_checks import run_integrity_checks
from src.model import run_model


def test_integrity_checks_pass_for_base_scenario(base_inputs):
    inputs = deepcopy(base_inputs)
    findings = run_integrity_checks(run_model(inputs), inputs, tol=1e-5)
    assert findings == []


def test_integrity_checks_pass_for_representative_scenarios(base_inputs):
    scenarios = [
        {"enable_term_loan": False, "enable_loc": False, "enable_distributions": False},
        {
            "new_build_mode": "schedule",
            "res_new_build_install_schedule": [{"month": "2026-01", "installs": 10.0}, {"month": "2026-02", "installs": 5.0}],
            "lc_new_build_install_schedule": [{"month": "2026-01", "installs": 2.0}],
        },
        {"new_build_mode": "annual_total", "res_new_build_annual_installs": 48.0, "lc_new_build_annual_installs": 12.0},
        {
            "asset_expiry_mode": "salvage",
            "asset_reuse_lag_months": 1,
            "tech_staffing_events": [{"month": "2026-01", "hires": 0, "attrition": 2}, {"month": "2026-02", "hires": 1, "attrition": 0}],
        },
        {"paid_leads_mode": "fixed", "paid_leads_per_month": 120, "enable_maintenance": False},
    ]
    for updates in scenarios:
        inputs = deepcopy(base_inputs)
        inputs.update(updates)
        findings = run_integrity_checks(run_model(inputs), inputs, tol=1e-5)
        assert findings == [], f"Unexpected integrity findings for updates={updates}: {findings}"


def test_integrity_checks_detects_identity_break(base_inputs):
    inputs = deepcopy(base_inputs)
    df = run_model(inputs)
    broken = df.copy()
    broken.loc[broken.index[0], "Total Revenue"] += 1.0
    findings = run_integrity_checks(broken, inputs, tol=1e-6)
    check_names = {f["Check"] for f in findings}
    assert "Revenue identity" in check_names
