"""LLM client + MockLLMClient for deterministic tests."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import AsyncIterator, Protocol, runtime_checkable

log = logging.getLogger("ace_buddy.llm")


@runtime_checkable
class LLMClient(Protocol):
    async def stream(self, system: str, user: str) -> AsyncIterator[str]: ...


class OpenAILLMClient:
    def __init__(self, model: str = "gpt-4o", cost_bound_usd: float = 10.0):
        self.model = model
        self.cost_bound_usd = cost_bound_usd
        self.total_cost_usd = 0.0
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI()
        except Exception as e:
            log.warning("AsyncOpenAI init failed: %s", e)
            self._client = None

    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        if self.total_cost_usd > self.cost_bound_usd:
            raise RuntimeError(
                f"cost bound exceeded: ${self.total_cost_usd:.3f} > ${self.cost_bound_usd:.3f}"
            )
        if self._client is None:
            yield "[llm unavailable]"
            return
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=True,
                temperature=0.2,
                max_tokens=220,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if text:
                    yield text
            # Cost accounting is rough — GPT-4o: $2.50/1M input (cache hit: 25%), $10/1M output
            # We don't get exact token counts in the stream; assume 2000 input, 150 output worst-case
            self.total_cost_usd += 0.0025 + 0.0015  # ~$0.004 per call, conservative
        except Exception as e:
            log.warning("openai stream failed: %s", e)
            yield f"[error: {e}]"


class MockLLMClient:
    """Emits canned tokens for deterministic tests. Reads MOCK_ANSWER env var as JSON list."""

    def __init__(self, tokens: list[str] | None = None, delay_s: float = 0.01):
        if tokens is None:
            raw = os.environ.get("MOCK_ANSWER")
            if raw:
                try:
                    tokens = json.loads(raw)
                except json.JSONDecodeError:
                    tokens = [raw]
            else:
                tokens = [
                    "**TL;DR**: ", "This is a mock answer for tests.", "\n",
                    "- ", "first anchor point\n",
                    "- ", "second anchor point\n",
                    "- ", "third anchor point\n",
                    "**Data**: ", "42% improvement.",
                ]
        self.tokens = list(tokens)
        self.delay_s = delay_s

    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        for tok in self.tokens:
            await asyncio.sleep(self.delay_s)
            yield tok
