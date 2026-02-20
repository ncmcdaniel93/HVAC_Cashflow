from __future__ import annotations

from copy import deepcopy

from src.defaults import DEFAULTS
from src.schema import migrate_assumptions
from src.sensitivity import available_sensitivity_drivers


def test_migration_from_v1_fields_generates_events():
    legacy = {
        "start_year": 2026,
        "start_month": 1,
        "horizon_months": 12,
        "starting_techs": 5,
        "tech_hire_per_quarter": 2,
        "work_days_per_month": 22,
        "avg_hours_per_tech_per_month": 160,
        "agreements_start": 300,
        "new_agreements_per_month": 15,
        "churn_annual_pct": 0.1,
        "maint_monthly_fee": 25.0,
    }
    migrated, warnings, unknown = migrate_assumptions(legacy)
    assert len(unknown) == 0
    assert migrated["res_agreements_start"] == 300
    assert migrated["res_new_agreements_per_month"] == 15
    assert migrated["res_churn_annual_pct"] == 0.1
    assert migrated["res_maint_monthly_fee"] == 25.0
    assert len(migrated["tech_staffing_events"]) >= 3
    assert any("deprecated" in w.lower() for w in warnings)
    assert any("migrated legacy tech_hire_per_quarter" in w.lower() for w in warnings)


def test_migration_clamps_percentages():
    payload = {"repl_close_rate": 2.5, "monthly_price_growth": -1.0, "value_mode": "invalid"}
    migrated, warnings, _ = migrate_assumptions(payload)
    assert migrated["repl_close_rate"] == 1.0
    assert migrated["monthly_price_growth"] == 0.0
    assert migrated["value_mode"] == "nominal"
    assert any("value_mode invalid" in w for w in warnings)


def test_migration_clamps_management_start_dates():
    payload = {
        "manager_start_year": 99999,
        "manager_start_month": -3,
        "ops_manager_start_year": 1900,
        "ops_manager_start_month": 18,
        "marketing_manager_start_year": 2500,
        "marketing_manager_start_month": 0,
    }
    migrated, _, _ = migrate_assumptions(payload)
    assert migrated["manager_start_year"] == 2100
    assert migrated["manager_start_month"] == 1
    assert migrated["ops_manager_start_year"] == 2000
    assert migrated["ops_manager_start_month"] == 12
    assert migrated["marketing_manager_start_year"] == 2100
    assert migrated["marketing_manager_start_month"] == 1


def test_migration_preserves_boolean_types_and_excludes_binary_toggles_from_sensitivity():
    payload = deepcopy(DEFAULTS)
    payload["enable_maintenance"] = "true"
    payload["enable_term_loan"] = 0
    payload["enable_loc"] = 1
    payload["enable_distributions"] = "false"
    migrated, _, _ = migrate_assumptions(payload)

    assert isinstance(migrated["enable_maintenance"], bool)
    assert isinstance(migrated["enable_term_loan"], bool)
    assert isinstance(migrated["enable_loc"], bool)
    assert isinstance(migrated["enable_distributions"], bool)
    assert migrated["enable_maintenance"] is True
    assert migrated["enable_term_loan"] is False
    assert migrated["enable_loc"] is True
    assert migrated["enable_distributions"] is False

    drivers = available_sensitivity_drivers(migrated)
    assert "enable_maintenance" not in drivers
    assert "enable_term_loan" not in drivers
    assert "enable_loc" not in drivers
    assert "enable_distributions" not in drivers
