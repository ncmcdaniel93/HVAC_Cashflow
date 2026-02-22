from __future__ import annotations

from pathlib import Path

import src.change_requests as change_requests


def test_append_change_request_requires_title_and_details(tmp_path, monkeypatch):
    log_dir = tmp_path / ".local_store"
    log_file = log_dir / "change_requests.jsonl"
    monkeypatch.setattr(change_requests, "LOG_DIR", log_dir)
    monkeypatch.setattr(change_requests, "CHANGE_REQUESTS_LOG_FILE", log_file)

    ok, message, request_id = change_requests.append_change_request(
        title="",
        details="Some detail",
        category="UX/UI",
        priority="Medium",
    )
    assert not ok
    assert "Title is required" in message
    assert request_id == ""

    ok, message, request_id = change_requests.append_change_request(
        title="Need better chart filters",
        details="",
        category="UX/UI",
        priority="Medium",
    )
    assert not ok
    assert "Details is required" not in message  # ensure message wording remains explicit
    assert "Details are required" in message
    assert request_id == ""


def test_append_and_read_change_requests(tmp_path, monkeypatch):
    log_dir = tmp_path / ".local_store"
    log_file = log_dir / "change_requests.jsonl"
    monkeypatch.setattr(change_requests, "LOG_DIR", log_dir)
    monkeypatch.setattr(change_requests, "CHANGE_REQUESTS_LOG_FILE", log_file)

    ok, message, request_id = change_requests.append_change_request(
        title="Add scenario-level assumptions diff view",
        details="Need a side-by-side before/after change summary for reviews.",
        category="Workflow/Productivity",
        priority="High",
        expected_outcome="Faster review and approvals",
        context={"scenario_name": "Base"},
    )
    assert ok
    assert "logged" in message.lower()
    assert request_id.startswith("CR-")
    assert Path(change_requests.change_request_log_path()).name == "change_requests.jsonl"

    rows = change_requests.read_change_requests(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["request_id"] == request_id
    assert row["category"] == "Workflow/Productivity"
    assert row["priority"] == "High"
    assert row["status"] == "new"


def test_configure_change_request_root_updates_paths(tmp_path):
    configured = change_requests.configure_change_request_root(tmp_path / "feedback_store")
    assert configured == tmp_path / "feedback_store"
    assert change_requests.LOG_DIR == tmp_path / "feedback_store"
    assert change_requests.CHANGE_REQUESTS_LOG_FILE == (tmp_path / "feedback_store" / "change_requests.jsonl")
