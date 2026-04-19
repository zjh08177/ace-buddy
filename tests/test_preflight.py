"""S6 tests: pre-flight individual checks."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ace_buddy.preflight import (
    check_config_files,
    check_mic,
    check_openai_key,
    check_screencapture,
    check_vision_import,
    run_preflight,
)


def test_check_screencapture():
    ok, msg = check_screencapture()
    assert ok, f"screencapture missing: {msg}"


def test_check_vision_import():
    ok, msg = check_vision_import()
    assert ok, f"Vision import failed: {msg}"


def test_check_openai_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ok, msg = check_openai_key()
    assert not ok
    assert "not set" in msg


def test_check_openai_key_wrong_prefix(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "wrongformat")
    ok, msg = check_openai_key()
    assert not ok
    assert "wrong prefix" in msg


def test_check_openai_key_valid_format(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abcdef1234567890")
    ok, msg = check_openai_key()
    assert ok
    assert "7890" in msg


def test_check_config_files_missing(tmp_path):
    ok, msg = check_config_files(tmp_path)
    assert not ok
    assert "missing" in msg


def test_check_config_files_present(tmp_path):
    for f in ("resume.md", "job.md", "system.md"):
        (tmp_path / f).write_text("x")
    ok, msg = check_config_files(tmp_path)
    assert ok


def test_run_preflight_reports_all_checks(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abcdef1234567890")
    for f in ("resume.md", "job.md", "system.md"):
        (tmp_path / f).write_text("x")
    r = run_preflight(tmp_path)
    assert len(r.checks) >= 4
    rep = r.report()
    assert "screencapture" in rep.lower() or "binary" in rep.lower()


def test_preflight_result_ok_is_and():
    from ace_buddy.preflight import PreflightResult
    r = PreflightResult()
    r.add("a", True)
    r.add("b", True)
    assert r.ok
    r.add("c", False)
    assert not r.ok
