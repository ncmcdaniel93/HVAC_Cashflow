from __future__ import annotations

from pathlib import Path

import src.runtime_logging as runtime_logging


def test_runtime_logging_append_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_logging, "LOG_DIR", Path(tmp_path))
    monkeypatch.setattr(runtime_logging, "RUNTIME_EVENTS_LOG_FILE", Path(tmp_path) / "runtime_events.jsonl")

    runtime_logging.append_runtime_event(
        level="warning",
        event="test_event",
        message="Test warning.",
        context={"case": "append_and_read"},
    )
    events = runtime_logging.read_runtime_events(limit=10)
    assert len(events) == 1
    assert events[0]["event"] == "test_event"
    assert events[0]["level"] == "WARNING"
    assert events[0]["context"]["case"] == "append_and_read"


def test_runtime_logging_handles_malformed_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_logging, "LOG_DIR", Path(tmp_path))
    log_file = Path(tmp_path) / "runtime_events.jsonl"
    monkeypatch.setattr(runtime_logging, "RUNTIME_EVENTS_LOG_FILE", log_file)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text('{"event":"ok","level":"INFO","timestamp_utc":"2026-01-01T00:00:00+00:00","message":"ok","context":{}}\nnot-json\n', encoding="utf-8")

    events = runtime_logging.read_runtime_events(limit=10)
    assert len(events) == 2
    assert events[0]["event"] == "ok"
    assert events[1]["event"] == "log_parse_error"
