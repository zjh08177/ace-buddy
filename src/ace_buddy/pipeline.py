"""Pipeline: on fire, grab audio + screen in parallel, call LLM, broadcast tokens."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable, Optional

from ace_buddy.audio import AudioSource, transcribe_wav_bytes
from ace_buddy.llm import LLMClient
from ace_buddy.prompt import PromptBuilder
from ace_buddy.server import ServerState, broadcast
from ace_buddy.vision import ScreenSource

log = logging.getLogger("ace_buddy.pipeline")

DEFAULT_AUDIO_WINDOW_SEC = 20.0
DEBOUNCE_SEC = 2.0


class Pipeline:
    """Owns the answer-generation loop. Serialized by a single asyncio.Lock."""

    def __init__(
        self,
        *,
        state: ServerState,
        audio: AudioSource,
        screen: ScreenSource,
        llm: LLMClient,
        prompt_builder: PromptBuilder,
        audio_window_sec: float = DEFAULT_AUDIO_WINDOW_SEC,
        whisper_client=None,  # inject for tests
        transcribe_fn=None,    # inject for full mock
    ):
        self.state = state
        self.audio = audio
        self.screen = screen
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.audio_window_sec = audio_window_sec
        self.whisper_client = whisper_client
        self.transcribe_fn = transcribe_fn or transcribe_wav_bytes
        self._lock = asyncio.Lock()
        self._last_fire_ts = 0.0
        self._next_answer_id = 1
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def fire_from_any_thread(self) -> None:
        """Thread-safe entry point called by hotkey C-thread or HTTP handler."""
        if self._loop is None:
            log.warning("pipeline not attached to loop, ignoring fire")
            return
        asyncio.run_coroutine_threadsafe(self.handle_answer_request(), self._loop)

    async def handle_answer_request(self) -> None:
        now = time.monotonic()
        if now - self._last_fire_ts < DEBOUNCE_SEC:
            log.info("debounced: %.2fs since last fire", now - self._last_fire_ts)
            return
        self._last_fire_ts = now

        if self._lock.locked():
            log.info("pipeline busy, skipping")
            return

        async with self._lock:
            answer_id = self._next_answer_id
            self._next_answer_id += 1
            self.state.current_answer.reset(answer_id)

            t0 = time.monotonic()
            await broadcast(self.state, {"type": "answer_start", "id": answer_id})

            # Sensors in parallel
            wav_bytes = self.audio.get_last_n_seconds(self.audio_window_sec)

            async def _do_stt():
                if not wav_bytes:
                    return ""
                return await self.transcribe_fn(
                    wav_bytes, client=self.whisper_client
                )

            async def _do_ocr():
                try:
                    return await self.screen.capture_and_ocr()
                except Exception as e:
                    log.warning("OCR failed: %s", e)
                    return ""

            try:
                transcript, ocr_text = await asyncio.gather(_do_stt(), _do_ocr())
            except Exception as e:
                log.exception("sensor gather failed: %s", e)
                transcript, ocr_text = "", ""

            sensor_ms = (time.monotonic() - t0) * 1000
            log.info(
                "sensors done in %.0fms — transcript=%d chars, ocr=%d chars",
                sensor_ms,
                len(transcript or ""),
                len(ocr_text or ""),
            )
            if not transcript and not ocr_text:
                log.warning(
                    "BOTH sensors returned empty! Check: "
                    "(1) mic permission granted? "
                    "(2) Screen Recording permission granted? "
                    "(3) restart app after granting permissions. "
                    "Visit /debug/sensors in browser to diagnose."
                )

            system_prompt, user_msg = self.prompt_builder.build(transcript, ocr_text)

            first_token_ts: Optional[float] = None
            full_text_parts: list[str] = []
            try:
                async for token in self.llm.stream(system_prompt, user_msg):
                    if first_token_ts is None:
                        first_token_ts = time.monotonic()
                        log.info(
                            "first token at %.0fms after fire",
                            (first_token_ts - t0) * 1000,
                        )
                    self.state.current_answer.append(token)
                    full_text_parts.append(token)
                    await broadcast(
                        self.state,
                        {"type": "answer_token", "id": answer_id, "text": token},
                    )
            except Exception as e:
                log.exception("llm stream error: %s", e)
                await broadcast(
                    self.state,
                    {"type": "error", "text": f"LLM error: {e}"},
                )
            finally:
                self.state.current_answer.finish()
                total_ms = (time.monotonic() - t0) * 1000
                await broadcast(
                    self.state,
                    {
                        "type": "answer_complete",
                        "id": answer_id,
                        "latency_ms": int(total_ms),
                        "transcript_chars": len(transcript or ""),
                        "ocr_chars": len(ocr_text or ""),
                    },
                )
                log.info(
                    "answer %d complete in %.0fms (%d chars)",
                    answer_id,
                    total_ms,
                    sum(len(p) for p in full_text_parts),
                )
