"""Anticipated-questions cheat sheet, pre-computed at boot."""
from __future__ import annotations

import json
import logging
from typing import Optional

from ace_buddy.prompt import ContextBundle

log = logging.getLogger("ace_buddy.cheatsheet")


CHEATSHEET_SYSTEM = (
    "You generate a list of 8 anticipated interview questions with "
    "terse bullet-style answer scaffolds, based on the candidate's resume "
    "and target role. Output strict JSON: "
    '{"questions": [{"q": "...", "a": "- anchor1\\n- anchor2\\n- anchor3"}, ...]}'
)


async def compute(ctx: ContextBundle, client=None, model: str = "gpt-4o") -> list[dict]:
    """Compute anticipated questions. Returns empty list on failure."""
    if client is None:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
        except Exception as e:
            log.warning("openai init failed: %s", e)
            return []

    user = (
        f"Resume:\n{ctx.resume_md.strip()}\n\n"
        f"Target role:\n{ctx.job_md.strip()}\n\n"
        "Generate 8 anticipated interview questions mixed: behavioral, "
        "technical, leadership, curveball. Keep answer scaffolds terse."
    )

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CHEATSHEET_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=1200,
        )
        content = resp.choices[0].message.content
        data = json.loads(content or "{}")
        qs = data.get("questions", [])
        if not isinstance(qs, list):
            return []
        cleaned = []
        for q in qs[:12]:
            if not isinstance(q, dict):
                continue
            cleaned.append({
                "q": str(q.get("q", "")).strip(),
                "a": str(q.get("a", "")).strip(),
            })
        return [c for c in cleaned if c["q"]]
    except Exception as e:
        log.warning("cheatsheet compute failed: %s", e)
        return []
