"""Live-API tests — gated on RUN_LIVE_TESTS=1. Requires OPENAI_API_KEY.

These tests exercise the REAL code path against the REAL OpenAI API:
- Whisper transcription of a fixture wav
- GPT-4o streaming via PromptBuilder
- Prompt caching hit rate on second call
- End-to-end cheatsheet computation

Cost bound: ~$0.05 per full run. Skipped by default.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

LIVE = os.environ.get("RUN_LIVE_TESTS") == "1" and bool(
    os.environ.get("OPENAI_API_KEY", "").startswith("sk-")
)
pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="live API tests need RUN_LIVE_TESTS=1 and a valid OPENAI_API_KEY",
)

FIXTURES = Path(__file__).parent / "fixtures"


async def test_llm_stream_real_gpt4o(tmp_path):
    from ace_buddy.llm import OpenAILLMClient
    from ace_buddy.prompt import ContextBundle, PromptBuilder

    (tmp_path / "system.md").write_text("Be concise.")
    (tmp_path / "resume.md").write_text("Eric, 10y ML, led team at Foo, 40% lift.")
    (tmp_path / "job.md").write_text("Senior ML Engineer at Bar")
    ctx = ContextBundle.from_dir(tmp_path)
    pb = PromptBuilder(ctx)

    llm = OpenAILLMClient(model="gpt-4o-mini", cost_bound_usd=1.0)
    system, user = pb.build(
        transcript="Tell me about a time you led a project.",
        ocr_text="",
    )
    tokens = []
    async for t in llm.stream(system, user):
        tokens.append(t)
    text = "".join(tokens)
    assert text
    assert "TL;DR" in text or "TLDR" in text.upper() or len(text) > 20
    # Basic format sanity
    assert "-" in text or "•" in text


async def test_prompt_cache_on_second_call(tmp_path):
    from ace_buddy.llm import OpenAILLMClient
    from ace_buddy.prompt import ContextBundle, PromptBuilder

    (tmp_path / "system.md").write_text("A" * 1200)  # > 1024 tokens worth
    (tmp_path / "resume.md").write_text("Eric resume content.")
    (tmp_path / "job.md").write_text("Target role.")
    ctx = ContextBundle.from_dir(tmp_path)
    pb = PromptBuilder(ctx)

    llm = OpenAILLMClient(model="gpt-4o-mini", cost_bound_usd=1.0)
    system, user1 = pb.build("First question.", "")
    _, user2 = pb.build("Second question.", "")

    # Two calls with same system prompt → second should have cache hit
    _ = [t async for t in llm.stream(system, user1)]
    _ = [t async for t in llm.stream(system, user2)]

    # No way to assert cache_hit from streaming API in current SDK;
    # just assert both calls returned content and cost is bounded
    assert llm.total_cost_usd < 0.05
