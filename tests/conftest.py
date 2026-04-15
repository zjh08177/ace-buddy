"""Test harness: AppHandle for spawning the app in e2e tests, plus fixtures."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class AppHandle:
    proc: subprocess.Popen
    port: int
    url: str

    def wait_ready(self, timeout: float = 8.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = httpx.get(f"{self.url}/status", timeout=0.5)
                if r.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.ReadTimeout):
                time.sleep(0.1)
            if self.proc.poll() is not None:
                out = (self.proc.stdout.read() if self.proc.stdout else b"").decode(errors="replace")
                raise RuntimeError(f"app exited early: code={self.proc.returncode} stdout={out}")
        raise TimeoutError(f"app did not become ready in {timeout}s")

    def terminate(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    @classmethod
    def spawn(
        cls,
        *,
        port: Optional[int] = None,
        no_auth: bool = True,
        headless: bool = True,
        extra_args: Optional[list[str]] = None,
        env_extra: Optional[dict] = None,
    ) -> "AppHandle":
        p = port or _free_port()
        args = [sys.executable, "-m", "ace_buddy.app", "--port", str(p)]
        if no_auth:
            args.append("--no-auth")
        if headless:
            args.append("--headless")
        if extra_args:
            args.extend(extra_args)

        env = os.environ.copy()
        env.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
        env.setdefault("ACE_BUDDY_CONFIG_DIR", str(REPO_ROOT / ".tmp-cfg"))
        if env_extra:
            env.update(env_extra)

        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(REPO_ROOT),
        )
        url = f"http://127.0.0.1:{p}"
        handle = cls(proc=proc, port=p, url=url)
        try:
            handle.wait_ready()
        except Exception:
            handle.terminate()
            raise
        return handle


@pytest.fixture
def app_handle():
    handle = AppHandle.spawn()
    try:
        yield handle
    finally:
        handle.terminate()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    d = REPO_ROOT / "tests" / "fixtures"
    d.mkdir(parents=True, exist_ok=True)
    return d
