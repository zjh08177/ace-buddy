"""Pre-flight checks: verify mic, screencapture, Vision, OpenAI reachable.

Fails loudly before any interview use. Returns (ok, messages) tuple.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

log = logging.getLogger("ace_buddy.preflight")


class PreflightResult:
    def __init__(self):
        self.checks: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, msg: str = "") -> None:
        self.checks.append((name, ok, msg))

    @property
    def ok(self) -> bool:
        return all(c[1] for c in self.checks)

    def report(self) -> str:
        lines = []
        for name, ok, msg in self.checks:
            marker = "✓" if ok else "✗"
            line = f"  {marker} {name}"
            if msg:
                line += f" — {msg}"
            lines.append(line)
        return "\n".join(lines)


def check_mic() -> tuple[bool, str]:
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_in = sd.default.device[0] if sd.default.device else None
        if default_in is None or default_in == -1:
            return False, "no default input device"
        d = devices[default_in] if isinstance(default_in, int) else default_in
        name = d.get("name", "?") if isinstance(d, dict) else str(d)
        return True, name
    except Exception as e:
        return False, f"sounddevice error: {e}"


def check_screencapture() -> tuple[bool, str]:
    path = shutil.which("screencapture") or "/usr/sbin/screencapture"
    if not Path(path).exists():
        return False, f"not found at {path}"
    return True, path


def check_vision_import() -> tuple[bool, str]:
    try:
        import Vision  # noqa: F401
        import Quartz  # noqa: F401
        return True, "pyobjc Vision+Quartz"
    except ImportError as e:
        return False, f"pyobjc import failed: {e}"


def check_openai_key() -> tuple[bool, str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return False, "OPENAI_API_KEY not set"
    if not key.startswith("sk-"):
        return False, "OPENAI_API_KEY has wrong prefix"
    return True, f"...{key[-6:]}"


def check_config_files(config_dir: Path) -> tuple[bool, str]:
    missing = [
        f for f in ("resume.md", "job.md", "system.md")
        if not (config_dir / f).exists()
    ]
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, str(config_dir)


def run_preflight(config_dir: Path, *, skip_network: bool = False) -> PreflightResult:
    r = PreflightResult()
    r.add("screencapture binary", *check_screencapture())
    r.add("pyobjc Vision+Quartz", *check_vision_import())
    r.add("microphone device", *check_mic())
    r.add("OPENAI_API_KEY", *check_openai_key())
    r.add("config files", *check_config_files(config_dir))
    return r
