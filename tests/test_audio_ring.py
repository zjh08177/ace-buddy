"""S3 tests: ring buffer + wav encoding + fixture source."""
from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from ace_buddy.audio import (
    FixtureAudioSource,
    RingBuffer,
    SAMPLE_RATE,
    pcm_to_wav_bytes,
)


def test_ring_buffer_writes_and_reads():
    rb = RingBuffer(capacity_samples=10)
    rb.write(np.array([1, 2, 3], dtype=np.int16))
    out = rb.get_last_n_samples(3)
    assert list(out) == [1, 2, 3]


def test_ring_buffer_wraparound():
    rb = RingBuffer(capacity_samples=5)
    rb.write(np.array([1, 2, 3, 4, 5], dtype=np.int16))
    rb.write(np.array([6, 7, 8], dtype=np.int16))  # wraps: buffer should be [6,7,8,4,5] in slots
    out = rb.get_last_n_samples(5)
    assert list(out) == [4, 5, 6, 7, 8]


def test_ring_buffer_evicts_old_samples():
    rb = RingBuffer(capacity_samples=4)
    rb.write(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int16))
    # newest 4 samples are 5,6,7,8
    out = rb.get_last_n_samples(4)
    assert list(out) == [5, 6, 7, 8]


def test_ring_buffer_partial_read():
    rb = RingBuffer(capacity_samples=10)
    rb.write(np.array([1, 2, 3], dtype=np.int16))
    out = rb.get_last_n_samples(2)
    assert list(out) == [2, 3]


def test_ring_buffer_read_more_than_written():
    rb = RingBuffer(capacity_samples=10)
    rb.write(np.array([1, 2, 3], dtype=np.int16))
    out = rb.get_last_n_samples(5)
    assert list(out) == [1, 2, 3]


def test_pcm_to_wav_bytes_roundtrip():
    pcm = np.array([0, 100, -100, 32767, -32768], dtype=np.int16)
    wav = pcm_to_wav_bytes(pcm)
    assert wav.startswith(b"RIFF")
    with wave.open(io.BytesIO(wav), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
        out = np.frombuffer(raw, dtype=np.int16)
        assert np.array_equal(out, pcm)


def test_pcm_empty_returns_empty():
    assert pcm_to_wav_bytes(np.zeros(0, dtype=np.int16)) == b""


def test_fixture_audio_source_empty_on_missing(tmp_path):
    src = FixtureAudioSource(tmp_path / "nope.wav")
    wav = src.get_last_n_seconds(5)
    assert wav == b""


def test_fixture_audio_source_returns_tail(tmp_path):
    # Create a 3-second synthetic wav
    samples = np.arange(SAMPLE_RATE * 3, dtype=np.int16) % 1000
    wav = pcm_to_wav_bytes(samples)
    p = tmp_path / "test.wav"
    p.write_bytes(wav)

    src = FixtureAudioSource(p)
    tail = src.get_last_n_seconds(1)
    with wave.open(io.BytesIO(tail), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE
        n = w.getnframes()
        # tail of 1 second, allow small tolerance
        assert SAMPLE_RATE - 10 <= n <= SAMPLE_RATE + 10
