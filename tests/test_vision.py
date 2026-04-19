"""S2 tests: Vision OCR + ScreencaptureScreenSource."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ace_buddy.vision import FixtureScreenSource, ScreencaptureScreenSource, ocr_png_bytes

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")

FIXTURES = Path(__file__).parent / "fixtures" / "screens"


async def test_fixture_source_returns_known_text():
    src = FixtureScreenSource(FIXTURES / "hello.png")
    text = await src.capture_and_ocr()
    assert text
    lowered = text.lower()
    assert "hello" in lowered
    assert "ace" in lowered or "buddy" in lowered


async def test_fixture_source_leetcode():
    src = FixtureScreenSource(FIXTURES / "leetcode_two_sum.png")
    text = await src.capture_and_ocr()
    assert text
    assert "two sum" in text.lower() or "indices" in text.lower()


async def test_paused_returns_empty():
    src = FixtureScreenSource(FIXTURES / "hello.png")
    src.paused = True
    text = await src.capture_and_ocr()
    assert text == ""


async def test_missing_file_returns_empty(tmp_path):
    src = FixtureScreenSource(tmp_path / "nope.png")
    text = await src.capture_and_ocr()
    assert text == ""


async def test_empty_bytes_ocr():
    text = await ocr_png_bytes(b"")
    assert text == ""


async def test_screencapture_source_paused():
    src = ScreencaptureScreenSource()
    src.paused = True
    text = await src.capture_and_ocr()
    assert text == ""


async def test_screencapture_bad_path_returns_empty():
    src = ScreencaptureScreenSource(screencapture_path="/nonexistent/screencapture")
    text = await src.capture_and_ocr()
    assert text == ""
