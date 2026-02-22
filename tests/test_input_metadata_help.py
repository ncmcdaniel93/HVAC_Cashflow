from __future__ import annotations

from src.input_metadata import calculation_logic_detail, help_with_guidance, impact_detail


def test_help_with_guidance_includes_calculation_and_impact_details():
    help_text = help_with_guidance("avg_service_ticket", "Average revenue per service visit.")
    assert "Reasonable range:" in help_text
    assert "Calculation use:" in help_text
    assert "Impact:" in help_text


def test_calculation_and_impact_detail_fallback_patterns():
    calc = calculation_logic_detail("custom_close_rate")
    impact = impact_detail("custom_close_rate")
    assert calc
    assert impact
