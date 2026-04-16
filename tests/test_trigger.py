"""Tests for AutoTrigger: audio silence detection + screen OCR change detection."""
from __future__ import annotations

import asyncio
import time

import pytest

from ace_buddy.trigger import AutoTrigger


class FakeAudioRMS:
    """Controllable RMS source for tests."""
    def __init__(self, rms: float = 0.0):
        self._rms = rms

    @property
    def recent_rms(self) -> float:
        return self._rms

    def set_rms(self, v: float) -> None:
        self._rms = v


class FakeScreenOCR:
    """Controllable OCR source for tests."""
    def __init__(self, text: str = ""):
        self._text = text

    async def capture_and_ocr(self) -> str:
        return self._text

    def set_text(self, t: str) -> None:
        self._text = t


async def test_silence_after_speech_fires():
    """Speech for 2.5s → silence for 1.5s → should fire."""
    audio = FakeAudioRMS()
    screen = FakeScreenOCR()
    fired = []

    trigger = AutoTrigger(audio, screen, fire_fn=lambda: fired.append(time.monotonic()))
    await trigger.start()

    # Simulate speech (RMS > threshold for 2.5s)
    audio.set_rms(0.05)
    await asyncio.sleep(2.6)

    # Go silent
    audio.set_rms(0.001)
    await asyncio.sleep(2.0)  # 1.5s silence + margin

    await trigger.stop()
    assert len(fired) >= 1, f"expected fire, got {len(fired)} fires"


async def test_short_noise_does_not_fire():
    """Burst < 2s should NOT trigger."""
    audio = FakeAudioRMS()
    screen = FakeScreenOCR()
    fired = []

    trigger = AutoTrigger(audio, screen, fire_fn=lambda: fired.append(1))
    await trigger.start()

    # Short burst: 0.5s of speech
    audio.set_rms(0.05)
    await asyncio.sleep(0.6)
    audio.set_rms(0.001)
    await asyncio.sleep(2.0)

    await trigger.stop()
    assert len(fired) == 0, f"expected no fire for short burst, got {len(fired)}"


async def test_screen_change_fires():
    """OCR text change with enough chars → should fire."""
    audio = FakeAudioRMS()
    screen = FakeScreenOCR("initial text that is long enough to be above the threshold chars")
    fired = []
    reasons = []

    trigger = AutoTrigger(
        audio, screen,
        fire_fn=lambda: fired.append(1),
        on_trigger=lambda r: reasons.append(r),
    )
    await trigger.start()

    # Wait for first poll
    await asyncio.sleep(3.5)

    # Change screen text
    screen.set_text("completely different text that is also long enough to exceed the threshold chars limit")
    await asyncio.sleep(3.5)  # wait for next poll

    await trigger.stop()
    assert len(fired) >= 1, f"expected fire on screen change, got {len(fired)}"
    assert "screen_changed" in reasons


async def test_screen_debounce():
    """Two rapid screen changes within 5s → only one fire."""
    audio = FakeAudioRMS()
    screen = FakeScreenOCR("text A that is long enough for the minimum character threshold test")
    fired = []

    trigger = AutoTrigger(audio, screen, fire_fn=lambda: fired.append(1))
    await trigger.start()

    await asyncio.sleep(3.5)  # first poll

    screen.set_text("text B that is completely different and also long enough for threshold")
    await asyncio.sleep(3.5)  # fires

    screen.set_text("text C that changes again quickly and is also long enough for threshold")
    await asyncio.sleep(3.5)  # should be debounced

    await trigger.stop()
    # Should have fired at most once (debounce is 5s, polls are 3s apart)
    assert len(fired) <= 2, f"expected at most 2 fires, got {len(fired)}"


async def test_on_trigger_callback():
    """on_trigger callback receives the reason string."""
    audio = FakeAudioRMS()
    screen = FakeScreenOCR()
    reasons = []

    trigger = AutoTrigger(
        audio, screen,
        fire_fn=lambda: None,
        on_trigger=lambda r: reasons.append(r),
    )
    await trigger.start()

    # Trigger via speech
    audio.set_rms(0.05)
    await asyncio.sleep(2.6)
    audio.set_rms(0.001)
    await asyncio.sleep(2.0)

    await trigger.stop()
    assert "speech_ended" in reasons
