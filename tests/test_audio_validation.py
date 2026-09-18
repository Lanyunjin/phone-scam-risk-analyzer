from __future__ import annotations

import wave

import pytest

from stt.audio_preprocessing import (
    AudioPreparationError,
    inspect_duration,
    validate_prepared_wav,
)


def _write_silent_wav(path, *, seconds: int, sample_rate: int = 16_000, channels: int = 1, sample_width: int = 2):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        frame = b"\x00" * sample_width * channels
        handle.writeframes(frame * sample_rate * seconds)


def test_accepts_short_synthetic_wav(tmp_path):
    path = tmp_path / "synthetic.wav"
    _write_silent_wav(path, seconds=1)
    assert inspect_duration(path) == pytest.approx(1.0)
    validate_prepared_wav(path)


def test_rejects_audio_over_five_minutes(tmp_path):
    path = tmp_path / "synthetic_long.wav"
    # A one-frame-per-second WAV keeps the temporary test file very small.
    _write_silent_wav(path, seconds=301, sample_rate=1)
    with pytest.raises(AudioPreparationError, match="300-second"):
        inspect_duration(path)


def test_rejects_wrong_prepared_format(tmp_path):
    path = tmp_path / "wrong_format.wav"
    _write_silent_wav(path, seconds=1, sample_rate=8_000)
    with pytest.raises(AudioPreparationError, match="16 kHz"):
        validate_prepared_wav(path)


def test_rejects_missing_input(tmp_path):
    with pytest.raises(AudioPreparationError, match="does not exist"):
        inspect_duration(tmp_path / "missing.wav")
