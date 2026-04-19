"""S4 tests: pipeline with mocked audio, screen, LLM."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

import pytest

from ace_buddy.llm import MockLLMClient
from ace_buddy.pipeline import Pipeline
from ace_buddy.prompt import ContextBundle, PromptBuilder
from ace_buddy.server import ServerState


class MockAudio:
    def __init__(self, wav: bytes = b"RIFF0000WAVEfmt ", transcript: str = "Tell me about yourself"):
        self._wav = wav
        self._transcript = transcript
        self.recent_rms = 0.0

    def start(self): pass
    def stop(self): pass
    def get_last_n_seconds(self, n: float) -> bytes:
        return self._wav


class MockScreen:
    def __init__(self, text: str = ""):
        self.text = text
        self.paused = False
    async def capture_and_ocr(self) -> str:
        return "" if self.paused else self.text
    async def aclose(self): pass


async def fake_transcribe(wav_bytes: bytes, *, client=None, **kwargs) -> str:
    if not wav_bytes:
        return ""
    return "Tell me about a time you handled conflict."


def _ctx(tmp_path) -> ContextBundle:
    (tmp_path / "system.md").write_text("Be concise.")
    (tmp_path / "resume.md").write_text("# Eric\n - Led ML team at Tencent, 2022")
    (tmp_path / "job.md").write_text("# Staff ML at Anthropic\nJD here.")
    return ContextBundle.from_dir(tmp_path)


def _state() -> ServerState:
    return ServerState(
        token="dev", host="127.0.0.1", port=0, config_dir=Path("/tmp"),
        auth_required=False, debug=False,
    )


async def test_pipeline_fires_and_tokens_buffered(tmp_path):
    ctx = _ctx(tmp_path)
    pb = PromptBuilder(ctx)
    state = _state()
    audio = MockAudio()
    screen = MockScreen(text="Two Sum")
    llm = MockLLMClient(tokens=["**TL;DR**: Hi\n", "- one\n", "- two\n", "- three\n", "**Data**: 42%"])

    pipe = Pipeline(
        state=state, audio=audio, screen=screen, llm=llm, prompt_builder=pb,
        transcribe_fn=fake_transcribe,
    )
    await pipe.handle_answer_request()

    ans = state.current_answer
    assert ans.complete
    assert ans.id == 1
    buffered = "".join(ans.tokens)
    assert "TL;DR" in buffered
    assert "42%" in buffered


async def test_pipeline_debounces_rapid_fires(tmp_path):
    ctx = _ctx(tmp_path)
    pb = PromptBuilder(ctx)
    state = _state()
    audio = MockAudio()
    screen = MockScreen()
    llm = MockLLMClient(tokens=["a"], delay_s=0)

    pipe = Pipeline(
        state=state, audio=audio, screen=screen, llm=llm, prompt_builder=pb,
        transcribe_fn=fake_transcribe,
    )
    await pipe.handle_answer_request()
    first_id = state.current_answer.id

    # Rapid second fire — should be debounced, answer id unchanged
    await pipe.handle_answer_request()
    assert state.current_answer.id == first_id


async def test_pipeline_ocr_only_path(tmp_path):
    """No audio transcript, only OCR text → still produces an answer."""
    ctx = _ctx(tmp_path)
    pb = PromptBuilder(ctx)
    state = _state()
    audio = MockAudio(wav=b"")  # no audio
    screen = MockScreen(text="Given an array of integers nums, return indices")
    llm = MockLLMClient(tokens=["**TL;DR**: Two Sum\n", "- hash map\n", "- O(n)\n", "- explain\n", "**Data**: "], delay_s=0)

    async def empty_transcribe(wav_bytes, **kwargs):
        return ""

    pipe = Pipeline(
        state=state, audio=audio, screen=screen, llm=llm, prompt_builder=pb,
        transcribe_fn=empty_transcribe,
    )
    await pipe.handle_answer_request()
    assert state.current_answer.complete
    buffered = "".join(state.current_answer.tokens)
    assert "Two Sum" in buffered


async def test_pipeline_paused_screen_empty_ocr(tmp_path):
    ctx = _ctx(tmp_path)
    pb = PromptBuilder(ctx)
    state = _state()
    screen = MockScreen(text="sensitive content")
    screen.paused = True
    llm = MockLLMClient(tokens=["x"], delay_s=0)

    pipe = Pipeline(
        state=state, audio=MockAudio(), screen=screen, llm=llm, prompt_builder=pb,
        transcribe_fn=fake_transcribe,
    )
    await pipe.handle_answer_request()
    # OCR returns empty when paused; pipeline still completes
    assert state.current_answer.complete
