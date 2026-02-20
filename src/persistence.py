"""Local persistence helpers for scenarios and workspaces."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from src.schema import SCHEMA_VERSION, SCENARIO_TYPE, WORKSPACE_TYPE, migrate_import_payload


STORE_DIR = Path(".local_store")
SCENARIO_STORE_FILE = STORE_DIR / "scenarios.json"
WORKSPACE_STORE_FILE = STORE_DIR / "workspaces.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_for(kind: str) -> Path:
    if kind == SCENARIO_TYPE:
        return SCENARIO_STORE_FILE
    if kind == WORKSPACE_TYPE:
        return WORKSPACE_STORE_FILE
    raise ValueError(f"Unsupported store kind: {kind}")


def _load_store(kind: str) -> dict:
    p = _path_for(kind)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_store(kind: str, data: dict) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    p = _path_for(kind)
    tmp = p.with_suffix(f"{p.suffix}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)


def list_saved_names(kind: str) -> list[str]:
    return sorted(_load_store(kind).keys())


def load_saved(kind: str, name: str) -> dict | None:
    return deepcopy(_load_store(kind).get(name))


def save_named_bundle(kind: str, name: str, bundle: dict, overwrite: bool = False) -> tuple[bool, str]:
    if not name.strip():
        return False, "Name is required."
    store = _load_store(kind)
    if name in store and not overwrite:
        return False, "Name already exists."
    store[name] = bundle
    _save_store(kind, store)
    return True, "Saved."


def delete_saved(kind: str, name: str) -> bool:
    store = _load_store(kind)
    if name not in store:
        return False
    del store[name]
    _save_store(kind, store)
    return True


def build_scenario_bundle(name: str, assumptions: dict) -> dict:
    return {
        "type": SCENARIO_TYPE,
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "created_at": _now_iso(),
        "assumptions": deepcopy(assumptions),
    }


def build_workspace_bundle(name: str, assumptions: dict, ui_state: dict) -> dict:
    return {
        "type": WORKSPACE_TYPE,
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "created_at": _now_iso(),
        "assumptions": deepcopy(assumptions),
        "ui_state": deepcopy(ui_state),
    }


def parse_import_json(raw_json: str) -> tuple[dict, dict | None, list[str], list[str]]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return {}, None, ["Could not parse import JSON."], []
    assumptions, ui_state, warnings, unknown_keys = migrate_import_payload(payload)
    return assumptions, ui_state, warnings, unknown_keys
