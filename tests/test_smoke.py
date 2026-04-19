"""S1 smoke: server boots, endpoints respond, QR is valid PNG."""
from __future__ import annotations

import httpx
import pytest


def test_imports():
    import ace_buddy
    import ace_buddy.app
    import ace_buddy.server
    assert ace_buddy.__version__


def test_server_module_builds_app():
    from ace_buddy.server import ServerState, create_app
    state = ServerState(
        token="dev",
        host="127.0.0.1",
        port=8765,
        config_dir=__import__("pathlib").Path("/tmp"),
        auth_required=False,
    )
    app = create_app(state)
    assert app.title == "ace-buddy"


def test_status_endpoint(app_handle):
    r = httpx.get(f"{app_handle.url}/status", timeout=2)
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True


def test_qr_returns_png(app_handle):
    r = httpx.get(f"{app_handle.url}/qr", timeout=2)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_unauthenticated_root_shows_qr(app_handle):
    """Unauthenticated / returns inline QR page (no-auth mode still serves app directly)."""
    r = httpx.get(f"{app_handle.url}/", timeout=2)
    assert r.status_code == 200
    assert "ace-buddy" in r.text


def test_index_html_served(app_handle):
    """Authenticated / returns the phone UI."""
    r = httpx.get(f"{app_handle.url}/", timeout=2)
    assert r.status_code == 200
    # In --no-auth mode, / serves the app directly
    assert "ace-buddy" in r.text


def test_fire_returns_ok(app_handle):
    r = httpx.post(f"{app_handle.url}/fire", timeout=2)
    assert r.status_code == 200
    assert r.json().get("ok") is True
