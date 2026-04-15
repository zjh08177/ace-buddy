"""S7 full-loop E2E: spawn app headless with mock LLM, fire /fire, assert tokens arrive on WS."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")

MOCK_TOKENS = [
    "**TL;DR**: ",
    "Led a cross-functional team through ambiguity by aligning stakeholders early.",
    "\n- clarified goals with PM + design\n",
    "- built a 2-week roadmap\n",
    "- shipped v1 on schedule\n",
    "**Data**: 40% engagement lift.",
]


@pytest.fixture
def e2e_handle(tmp_path):
    """Spawn app in headless mock mode with fixture audio + screen."""
    from tests.conftest import AppHandle

    # Create fixture files
    fx_audio = tmp_path / "audio.wav"
    # minimal valid WAV: 0.5s of silence
    import wave
    with wave.open(str(fx_audio), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 8000)

    cfg_dir = tmp_path / ".ace-buddy"
    cfg_dir.mkdir()
    (cfg_dir / "resume.md").write_text("# Eric\n- Led ML team, 40% engagement lift\n")
    (cfg_dir / "job.md").write_text("# Senior ML Engineer at Foo\nLooking for leadership\n")
    (cfg_dir / "system.md").write_text("Be concise.")

    handle = AppHandle.spawn(
        extra_args=[
            "--fixture-audio", str(fx_audio),
            "--fixture-screen", "/nonexistent.png",  # empty OCR
            "--mock-llm", json.dumps(MOCK_TOKENS),
        ],
        env_extra={
            "ACE_BUDDY_CONFIG_DIR": str(cfg_dir),
            "OPENAI_API_KEY": "sk-test-placeholder",
        },
    )
    try:
        yield handle
    finally:
        handle.terminate()


async def _connect_ws_and_collect(url: str, timeout_sec: float = 8.0) -> dict:
    """Connect to /ws, fire a trigger via POST /fire, collect all events until answer_complete."""
    import websockets

    ws_url = url.replace("http://", "ws://") + "/ws"
    events: list[dict] = []
    first_token_elapsed: float | None = None
    start = asyncio.get_running_loop().time()

    async with websockets.connect(ws_url) as ws:
        # Expect hello first
        hello_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        hello = json.loads(hello_raw)
        assert hello.get("type") == "hello"

        # Fire via POST /fire
        async with httpx.AsyncClient(timeout=2.0) as client:
            fire_r = await client.post(f"{url}/fire")
            assert fire_r.status_code == 200

        fire_ts = asyncio.get_running_loop().time()

        # Collect events until answer_complete
        deadline = asyncio.get_running_loop().time() + timeout_sec
        while asyncio.get_running_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(
                    ws.recv(),
                    timeout=max(0.1, deadline - asyncio.get_running_loop().time()),
                )
            except asyncio.TimeoutError:
                break
            msg = json.loads(raw)
            events.append(msg)
            if msg.get("type") == "answer_token" and first_token_elapsed is None:
                first_token_elapsed = asyncio.get_running_loop().time() - fire_ts
            if msg.get("type") == "answer_complete":
                break

    return {
        "events": events,
        "first_token_elapsed_sec": first_token_elapsed,
    }


async def test_full_loop_mock_tokens_arrive(e2e_handle):
    result = await _connect_ws_and_collect(e2e_handle.url, timeout_sec=10.0)
    events = result["events"]
    types = [e.get("type") for e in events]
    assert "answer_start" in types
    assert "answer_token" in types
    assert "answer_complete" in types

    text = "".join(
        e.get("text", "") for e in events if e.get("type") == "answer_token"
    )
    assert "TL;DR" in text
    assert "40% engagement lift" in text
    assert "- " in text

    # Latency assertion (generous because headless + cold start on first call)
    assert result["first_token_elapsed_sec"] is not None
    assert result["first_token_elapsed_sec"] < 5.0, (
        f"first token too slow: {result['first_token_elapsed_sec']:.2f}s"
    )


async def test_full_loop_answer_complete_has_latency(e2e_handle):
    result = await _connect_ws_and_collect(e2e_handle.url, timeout_sec=10.0)
    complete = next(
        (e for e in result["events"] if e.get("type") == "answer_complete"),
        None,
    )
    assert complete is not None
    assert "latency_ms" in complete
    assert isinstance(complete["latency_ms"], int)


async def test_phone_tap_via_ws_message(e2e_handle):
    """Send {type: 'fire'} over WS instead of POST /fire."""
    import websockets
    ws_url = e2e_handle.url.replace("http://", "ws://") + "/ws"
    async with websockets.connect(ws_url) as ws:
        hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert hello["type"] == "hello"
        await ws.send(json.dumps({"type": "fire"}))
        # Collect answer_start within 5s
        got_start = False
        deadline = asyncio.get_running_loop().time() + 5.0
        while asyncio.get_running_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            msg = json.loads(raw)
            if msg.get("type") == "answer_start":
                got_start = True
                break
        assert got_start
