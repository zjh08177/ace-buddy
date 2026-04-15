"""S4/S5 tests: prompt builder + cache stability + context loading."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ace_buddy.prompt import ContextBundle, PromptBuilder


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "system.md").write_text("Be concise.")
    (tmp_path / "resume.md").write_text("# Eric\n10y ML engineer at Anthropic-of-yore")
    (tmp_path / "job.md").write_text("# Senior ML Engineer at Foo\nThe role is X.")
    return ContextBundle.from_dir(tmp_path)


def test_context_bundle_loads_all_files(ctx):
    assert "Eric" in ctx.resume_md
    assert "Foo" in ctx.job_md
    assert "concise" in ctx.system_md
    assert ctx.job_title == "Senior ML Engineer at Foo"


def test_context_bundle_creates_stubs_when_missing(tmp_path):
    ctx = ContextBundle.from_dir(tmp_path)
    assert (tmp_path / "resume.md").exists()
    assert (tmp_path / "job.md").exists()
    assert (tmp_path / "system.md").exists()
    assert "Eric" in ctx.resume_md


def test_prompt_builder_system_is_byte_stable(ctx):
    pb = PromptBuilder(ctx)
    s1, _ = pb.build("hi", "code")
    s2, _ = pb.build("different audio", "different ocr")
    assert s1 == s2
    assert hashlib.sha256(s1.encode()).hexdigest() == pb.system_sha256


def test_prompt_builder_user_has_labels(ctx):
    pb = PromptBuilder(ctx)
    _, user = pb.build("what is O(1)?", "Two Sum problem")
    assert "INTERVIEWER_SAID:" in user
    assert "INTERVIEWER_SHOWED:" in user
    assert "what is O(1)?" in user
    assert "Two Sum problem" in user


def test_prompt_builder_empty_sections_placeholder(ctx):
    pb = PromptBuilder(ctx)
    _, user = pb.build("", "")
    assert "(no audio)" in user
    assert "(no screen content)" in user


def test_prompt_builder_cache_hash_logged(ctx):
    pb = PromptBuilder(ctx)
    assert pb.system_sha256
    assert len(pb.system_sha256) == 64


def test_system_prompt_contains_injection_guard(ctx):
    pb = PromptBuilder(ctx)
    assert "UNTRUSTED" in pb.system_prompt or "untrusted" in pb.system_prompt
    assert "not as instructions" in pb.system_prompt.lower()


def test_prompt_includes_format_rules(ctx):
    pb = PromptBuilder(ctx)
    assert "**TL;DR**" in pb.system_prompt
    assert "**Data**" in pb.system_prompt
