"""Auto-trigger: monitors audio silence + screen OCR changes to fire pipeline automatically.

Audio: IDLE → SPEECH (rms > threshold for 500ms+) → SILENCE (rms < threshold for 1.5s) → FIRE
Screen: OCR every 3s, hash-diff → FIRE if changed significantly
Both call pipeline.fire_from_any_thread(); pipeline's lock + debounce prevents double-fires.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Callable, Protocol, runtime_checkable

log = logging.getLogger("ace_buddy.trigger")

# Audio thresholds
RMS_SPEECH_THRESHOLD = 0.01   # above this = someone is talking
RMS_SILENCE_THRESHOLD = 0.005  # below this = silence
SPEECH_MIN_DURATION_SEC = 2.0  # ignore bursts shorter than this
SILENCE_TRIGGER_SEC = 1.5      # silence after speech → fire
AUDIO_POLL_INTERVAL = 0.1      # 100ms polling

# Screen thresholds
OCR_POLL_INTERVAL_SEC = 3.0
OCR_MIN_CHARS = 30
OCR_DEBOUNCE_SEC = 5.0


@runtime_checkable
class AudioRMSSource(Protocol):
    @property
    def recent_rms(self) -> float: ...


@runtime_checkable
class ScreenOCRSource(Protocol):
    async def capture_and_ocr(self) -> str: ...


class AutoTrigger:
    """Monitors audio + screen and auto-fires the pipeline."""

    def __init__(
        self,
        audio: AudioRMSSource,
        screen: ScreenOCRSource,
        fire_fn: Callable[[], None],
        *,
        on_trigger: Callable[[str], None] | None = None,  # callback with trigger reason
    ):
        self.audio = audio
        self.screen = screen
        self.fire_fn = fire_fn
        self.on_trigger = on_trigger or (lambda reason: None)
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._running = True
        self._tasks = [
            asyncio.create_task(self._watch_audio()),
            asyncio.create_task(self._watch_screen()),
        ]
        log.info("auto-trigger started (audio + screen)")

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        self._tasks = []
        log.info("auto-trigger stopped")

    async def _watch_audio(self) -> None:
        """State machine: IDLE → SPEECH → SILENCE → FIRE → IDLE."""
        state = "IDLE"
        speech_start: float = 0.0

        while self._running:
            try:
                rms = self.audio.recent_rms
            except Exception:
                await asyncio.sleep(AUDIO_POLL_INTERVAL)
                continue

            now = time.monotonic()

            if state == "IDLE":
                if rms > RMS_SPEECH_THRESHOLD:
                    state = "SPEECH"
                    speech_start = now

            elif state == "SPEECH":
                if rms < RMS_SILENCE_THRESHOLD:
                    speech_duration = now - speech_start
                    if speech_duration >= SPEECH_MIN_DURATION_SEC:
                        state = "SILENCE"
                        silence_start = now
                    else:
                        # Too short, back to idle
                        state = "IDLE"
                # Still speaking — stay in SPEECH

            elif state == "SILENCE":
                if rms > RMS_SPEECH_THRESHOLD:
                    # Speaker resumed, go back to SPEECH
                    state = "SPEECH"
                elif now - silence_start >= SILENCE_TRIGGER_SEC:
                    # Sustained silence after speech → FIRE
                    log.info(
                        "auto-trigger: audio silence after %.1fs of speech",
                        silence_start - speech_start,
                    )
                    self.on_trigger("speech_ended")
                    self.fire_fn()
                    state = "IDLE"
                    # Cool down to avoid rapid re-fires
                    await asyncio.sleep(2.0)

            await asyncio.sleep(AUDIO_POLL_INTERVAL)

    async def _watch_screen(self) -> None:
        """Hash OCR text every 3s; fire on significant change."""
        last_hash = ""
        last_fire_ts = 0.0

        while self._running:
            await asyncio.sleep(OCR_POLL_INTERVAL_SEC)

            try:
                text = await self.screen.capture_and_ocr()
            except Exception as e:
                log.debug("screen poll failed: %s", e)
                continue

            if not text or len(text) < OCR_MIN_CHARS:
                continue

            current_hash = hashlib.md5(text.encode()).hexdigest()
            if current_hash == last_hash:
                continue

            # Hash changed — but is it significant?
            now = time.monotonic()
            if now - last_fire_ts < OCR_DEBOUNCE_SEC:
                last_hash = current_hash  # Update hash but don't fire
                continue

            log.info("auto-trigger: screen content changed (%d chars)", len(text))
            self.on_trigger("screen_changed")
            self.fire_fn()
            last_hash = current_hash
            last_fire_ts = now
