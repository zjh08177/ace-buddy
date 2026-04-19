"""Microphone capture + Whisper non-streaming transcription.

- AudioSource Protocol: start/stop/get_last_n_seconds(n) → wav bytes
- SoundDeviceAudioSensor: real mic, 60s ring buffer at 16kHz mono
- FixtureAudioSource: reads a canned .wav file, returns last n seconds
- transcribe_wav_bytes(): calls OpenAI whisper-1 on wav bytes
"""
from __future__ import annotations

import asyncio
import io
import logging
import struct
import wave
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

log = logging.getLogger("ace_buddy.audio")

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
DTYPE_NP = np.int16
BYTES_PER_SAMPLE = 2
RING_BUFFER_SECONDS = 60
RING_BUFFER_SAMPLES = SAMPLE_RATE * RING_BUFFER_SECONDS


@runtime_checkable
class AudioSource(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_last_n_seconds(self, n: float) -> bytes: ...
    @property
    def recent_rms(self) -> float: ...


def pcm_to_wav_bytes(pcm: np.ndarray) -> bytes:
    """Wrap int16 PCM samples in a WAV container."""
    if pcm.size == 0:
        return b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(BYTES_PER_SAMPLE)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.astype(DTYPE_NP).tobytes())
    return buf.getvalue()


class RingBuffer:
    """Simple fixed-size int16 ring buffer. Thread-safe for a single writer + reader."""

    def __init__(self, capacity_samples: int = RING_BUFFER_SAMPLES):
        self.capacity = capacity_samples
        self.buf = np.zeros(capacity_samples, dtype=DTYPE_NP)
        self.write_pos = 0
        self.total_written = 0
        import threading
        self._lock = threading.Lock()

    def write(self, chunk: np.ndarray) -> None:
        with self._lock:
            n = chunk.size
            if n >= self.capacity:
                self.buf[:] = chunk[-self.capacity:]
                self.write_pos = 0
                self.total_written += n
                return
            end = self.write_pos + n
            if end <= self.capacity:
                self.buf[self.write_pos:end] = chunk
            else:
                split = self.capacity - self.write_pos
                self.buf[self.write_pos:] = chunk[:split]
                self.buf[:n - split] = chunk[split:]
            self.write_pos = end % self.capacity
            self.total_written += n

    def get_last_n_samples(self, n_samples: int) -> np.ndarray:
        with self._lock:
            available = min(self.total_written, self.capacity)
            n = min(n_samples, available)
            if n == 0:
                return np.zeros(0, dtype=DTYPE_NP)
            start = (self.write_pos - n) % self.capacity
            if start + n <= self.capacity:
                return self.buf[start:start + n].copy()
            return np.concatenate(
                (self.buf[start:], self.buf[: (start + n) % self.capacity])
            )


class SoundDeviceAudioSensor:
    """Wraps sounddevice InputStream; fills a 60s ring buffer."""

    def __init__(self, samplerate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.samplerate = samplerate
        self.channels = channels
        self.ring = RingBuffer()
        self._stream = None
        self._rms = 0.0

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.debug("sounddevice status: %s", status)
        # indata shape: (frames, channels) int16
        chunk = indata[:, 0] if indata.ndim == 2 else indata
        chunk = np.asarray(chunk, dtype=DTYPE_NP).reshape(-1)
        self.ring.write(chunk)
        if chunk.size:
            # Quick RMS for level indicator
            x = chunk.astype(np.float32) / 32768.0
            self._rms = float(np.sqrt(np.mean(x * x)))

    def start(self) -> None:
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=DTYPE,
            callback=self._callback,
        )
        self._stream.start()
        log.info("audio sensor started @ %sHz", self.samplerate)

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def get_last_n_seconds(self, n: float) -> bytes:
        pcm = self.ring.get_last_n_samples(int(n * self.samplerate))
        return pcm_to_wav_bytes(pcm)

    @property
    def recent_rms(self) -> float:
        return self._rms


class FixtureAudioSource:
    """Loads a wav file at construction; `get_last_n_seconds` returns the tail."""

    def __init__(self, wav_path: str | Path):
        self.wav_path = Path(wav_path)
        self._samples: np.ndarray = np.zeros(0, dtype=DTYPE_NP)
        self._load()

    def _load(self) -> None:
        if not self.wav_path.exists():
            return
        with wave.open(str(self.wav_path), "rb") as w:
            if w.getnchannels() not in (1, 2):
                return
            nframes = w.getnframes()
            sw = w.getsampwidth()
            raw = w.readframes(nframes)
            if sw == 2:
                arr = np.frombuffer(raw, dtype=np.int16)
            else:
                return
            if w.getnchannels() == 2:
                arr = arr.reshape(-1, 2).mean(axis=1).astype(np.int16)
            self._samples = arr

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_last_n_seconds(self, n: float) -> bytes:
        n_samples = int(n * SAMPLE_RATE)
        tail = self._samples[-n_samples:] if n_samples > 0 else self._samples
        return pcm_to_wav_bytes(tail)

    @property
    def recent_rms(self) -> float:
        return 0.0


async def transcribe_wav_bytes(
    wav_bytes: bytes,
    *,
    client=None,
    model: str = "whisper-1",
) -> str:
    """Call OpenAI Whisper with wav bytes. Returns transcript text or empty string."""
    if not wav_bytes or len(wav_bytes) < 100:
        return ""
    if client is None:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
        except Exception as e:
            log.warning("openai import failed: %s", e)
            return ""
    try:
        # Whisper expects a file-like. Use a NamedTemporaryFile or BytesIO with .name.
        buf = io.BytesIO(wav_bytes)
        buf.name = "audio.wav"
        resp = await client.audio.transcriptions.create(
            model=model,
            file=buf,
            response_format="text",
        )
        # response_format=text returns a string; json returns an object
        if isinstance(resp, str):
            return resp.strip()
        return str(getattr(resp, "text", "")).strip()
    except Exception as e:
        log.warning("whisper transcribe failed: %s", e)
        return ""
