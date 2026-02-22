"""Runtime diagnostics logging helpers for production support."""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from streamlit.runtime.scriptrunner import get_script_run_ctx


LOG_DIR = Path(".local_store")
RUNTIME_EVENTS_LOG_FILE = LOG_DIR / "runtime_events.jsonl"

_DEFAULT_LOG_DIR = Path(".local_store")
_STORAGE_ENV_VAR = "HVAC_STORAGE_ROOT"

_EXCEPTION_HOOK_INSTALLED = False


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


def configure_log_root(path_value: str | Path | None) -> Path:
    global LOG_DIR, RUNTIME_EVENTS_LOG_FILE
    LOG_DIR = _expand_log_root(path_value)
    RUNTIME_EVENTS_LOG_FILE = LOG_DIR / "runtime_events.jsonl"
    return LOG_DIR


def runtime_log_path() -> str:
    return str(RUNTIME_EVENTS_LOG_FILE.resolve())


def append_runtime_event(
    level: str,
    event: str,
    message: str,
    context: dict[str, Any] | None = None,
    exc: BaseException | None = None,
) -> None:
    """Append a structured runtime event record to disk."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "timestamp_utc": _now_iso(),
            "level": str(level).upper(),
            "event": str(event),
            "message": str(message),
            "context": context or {},
        }
        if exc is not None:
            record["exception_type"] = type(exc).__name__
            record["exception_message"] = str(exc)
            if exc.__traceback__ is not None:
                record["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            else:
                record["traceback"] = traceback.format_exc()
        with RUNTIME_EVENTS_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_safe_json_default, ensure_ascii=False) + "\n")
    except Exception:
        # Diagnostics should never crash the app.
        pass


def read_runtime_events(limit: int = 200) -> list[dict[str, Any]]:
    if limit <= 0 or not RUNTIME_EVENTS_LOG_FILE.exists():
        return []
    try:
        lines = RUNTIME_EVENTS_LOG_FILE.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-int(limit) :]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append(
                {
                    "timestamp_utc": _now_iso(),
                    "level": "ERROR",
                    "event": "log_parse_error",
                    "message": "Malformed log line encountered.",
                    "context": {"line": line},
                }
            )
    return out


def install_global_exception_logging() -> None:
    """Capture uncaught exceptions into runtime log."""
    global _EXCEPTION_HOOK_INSTALLED
    if _EXCEPTION_HOOK_INSTALLED:
        return
    old_hook = sys.excepthook

    def _hook(exc_type, exc, exc_tb):
        try:
            # Avoid polluting runtime diagnostics with non-Streamlit process exceptions
            # (for example, ad-hoc local scripts that import app modules).
            if get_script_run_ctx() is None:
                old_hook(exc_type, exc, exc_tb)
                return
            tb_text = "".join(traceback.format_exception(exc_type, exc, exc_tb))
            append_runtime_event(
                level="ERROR",
                event="uncaught_exception",
                message=str(exc),
                context={"traceback": tb_text},
                exc=exc,
            )
        except Exception:
            pass
        old_hook(exc_type, exc, exc_tb)

    sys.excepthook = _hook
    _EXCEPTION_HOOK_INSTALLED = True


configure_log_root(os.getenv(_STORAGE_ENV_VAR, ""))
