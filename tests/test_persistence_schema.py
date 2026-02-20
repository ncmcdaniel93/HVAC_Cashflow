from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from src.defaults import DEFAULTS
import src.persistence as persistence


def test_workspace_bundle_roundtrip_and_local_store(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "STORE_DIR", Path(tmp_path))
    monkeypatch.setattr(persistence, "SCENARIO_STORE_FILE", Path(tmp_path) / "scenarios.json")
    monkeypatch.setattr(persistence, "WORKSPACE_STORE_FILE", Path(tmp_path) / "workspaces.json")

    assumptions = deepcopy(DEFAULTS)
    ui_state = {"range_preset": "Full horizon", "value_mode": "nominal"}
    bundle = persistence.build_workspace_bundle("ws1", assumptions, ui_state)

    ok, _ = persistence.save_named_bundle("workspace", "ws1", bundle, overwrite=False)
    assert ok
    loaded = persistence.load_saved("workspace", "ws1")
    assert loaded is not None
    assert loaded["type"] == "workspace"
    assert loaded["name"] == "ws1"
    assert loaded["assumptions"]["start_year"] == assumptions["start_year"]
    assert loaded["ui_state"]["range_preset"] == "Full horizon"


def test_parse_import_json_supports_legacy_dict():
    legacy_json = json.dumps({"start_year": 2027, "start_month": 3, "horizon_months": 24})
    assumptions, ui_state, warnings, unknown = persistence.parse_import_json(legacy_json)
    assert assumptions["start_year"] == 2027
    assert assumptions["start_month"] == 3
    assert assumptions["horizon_months"] == 24
    assert ui_state is None
    assert len(unknown) == 0
    assert any("legacy" in w.lower() for w in warnings)

