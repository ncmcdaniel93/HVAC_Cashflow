"""Persistent change-request logging for enhancement backlog planning."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


LOG_DIR = Path(".local_store")
CHANGE_REQUESTS_LOG_FILE = LOG_DIR / "change_requests.jsonl"

_DEFAULT_LOG_DIR = Path(".local_store")
_STORAGE_ENV_VAR = "HVAC_STORAGE_ROOT"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_default(value: Any):
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _expand_log_root(path_value: str | Path | None) -> Path:
    if path_value is None:
        return _DEFAULT_LOG_DIR
    text = str(path_value).strip()
    if not text:
        return _DEFAULT_LOG_DIR
    expanded = os.path.expandvars(os.path.expanduser(text))
    return Path(expanded)


def configure_change_request_root(path_value: str | Path | None) -> Path:
    global LOG_DIR, CHANGE_REQUESTS_LOG_FILE
    LOG_DIR = _expand_log_root(path_value)
    CHANGE_REQUESTS_LOG_FILE = LOG_DIR / "change_requests.jsonl"
    return LOG_DIR


def change_request_log_path() -> str:
    return str(CHANGE_REQUESTS_LOG_FILE.resolve())


def append_change_request(
    *,
    title: str,
    details: str,
    category: str,
    priority: str,
    expected_outcome: str = "",
    context: dict[str, Any] | None = None,
) -> tuple[bool, str, str]:
    """Append one change-request record to local storage.

    Returns: (ok, message, request_id)
    """
    title_text = str(title or "").strip()
    details_text = str(details or "").strip()
    if not title_text:
        return False, "Title is required.", ""
    if not details_text:
        return False, "Details are required.", ""

    request_id = f"CR-{uuid4().hex[:10].upper()}"
    record = {
        "request_id": request_id,
        "timestamp_utc": _now_iso(),
        "title": title_text,
        "details": details_text,
        "category": str(category or "General").strip(),
        "priority": str(priority or "Medium").strip(),
        "expected_outcome": str(expected_outcome or "").strip(),
        "context": context or {},
        "status": "new",
    }

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with CHANGE_REQUESTS_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_safe_json_default) + "\n")
    except Exception as exc:
        return False, f"Failed to write change request log: {exc}", ""

    return True, "Change request logged.", request_id


def read_change_requests(limit: int = 100) -> list[dict[str, Any]]:
    if limit <= 0 or not CHANGE_REQUESTS_LOG_FILE.exists():
        return []
    try:
        lines = CHANGE_REQUESTS_LOG_FILE.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for line in lines[-int(limit) :]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append(
                {
                    "request_id": "",
                    "timestamp_utc": _now_iso(),
                    "title": "Malformed backlog log entry",
                    "details": line,
                    "category": "System",
                    "priority": "Low",
                    "expected_outcome": "",
                    "context": {},
                    "status": "error",
                }
            )
    return out


configure_change_request_root(os.getenv(_STORAGE_ENV_VAR, ""))
