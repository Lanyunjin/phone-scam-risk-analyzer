"""Validate and convert local audio for Azure Speech.

FFmpeg and ffprobe are external system dependencies. Commands use argument lists
and never invoke a shell. Source audio is never overwritten.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

MAX_AUDIO_SECONDS = 300.0
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a"}


class AudioPreparationError(RuntimeError):
    """Raised when audio validation or conversion cannot be completed safely."""


@dataclass
class PreparedAudio:
    path: Path
    source_duration_seconds: float
    _temporary_directory: tempfile.TemporaryDirectory[str] | None = field(
        default=None, repr=False
    )

    def cleanup(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __enter__(self) -> "PreparedAudio":
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()


def _require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise AudioPreparationError(
            f"Required external executable '{name}' was not found on PATH."
        )
    return executable


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            if frame_rate <= 0:
                raise AudioPreparationError("WAV file has an invalid frame rate.")
            return handle.getnframes() / float(frame_rate)
    except (wave.Error, EOFError) as exc:
        raise AudioPreparationError("Input WAV file is invalid or unreadable.") from exc


def _ffprobe_duration(path: Path) -> float:
    ffprobe = _require_executable("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
            timeout=30,
        )
        payload = json.loads(completed.stdout)
        return float(payload["format"]["duration"])
    except subprocess.TimeoutExpired as exc:
        raise AudioPreparationError("ffprobe timed out while inspecting audio.") from exc
    except (subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise AudioPreparationError("Unable to determine audio duration safely.") from exc


def inspect_duration(input_path: str | Path) -> float:
    path = Path(input_path).expanduser()
    if not path.is_file():
        raise AudioPreparationError("Input audio file does not exist or is not a file.")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AudioPreparationError(f"Unsupported audio type. Expected one of: {allowed}.")

    duration = _wav_duration(path) if path.suffix.lower() == ".wav" else _ffprobe_duration(path)
    if duration <= 0:
        raise AudioPreparationError("Audio duration must be greater than zero.")
    if duration > MAX_AUDIO_SECONDS:
        raise AudioPreparationError(
            f"Audio duration exceeds the {int(MAX_AUDIO_SECONDS)}-second limit."
        )
    return duration


def validate_prepared_wav(input_path: str | Path) -> None:
    """Require the exact WAV format expected by the public STT boundary."""

    path = Path(input_path).expanduser()
    if not path.is_file():
        raise AudioPreparationError("Prepared WAV file does not exist.")
    try:
        with wave.open(str(path), "rb") as handle:
            valid = (
                handle.getframerate() == 16_000
                and handle.getnchannels() == 1
                and handle.getsampwidth() == 2
                and handle.getcomptype() == "NONE"
            )
    except (wave.Error, EOFError) as exc:
        raise AudioPreparationError("Prepared WAV file is invalid.") from exc
    if not valid:
        raise AudioPreparationError(
            "Prepared audio must be 16 kHz, mono, PCM signed 16-bit WAV."
        )


def prepare_audio(
    input_path: str | Path, output_path: str | Path | None = None
) -> PreparedAudio:
    """Convert WAV/MP3/M4A to 16 kHz mono signed 16-bit PCM WAV."""

    source = Path(input_path).expanduser().resolve()
    duration = inspect_duration(source)
    ffmpeg = _require_executable("ffmpeg")
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None

    if output_path is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="phone_scam_audio_")
        destination = Path(temporary_directory.name) / "prepared.wav"
    else:
        destination = Path(output_path).expanduser().resolve()
        if destination == source:
            raise AudioPreparationError("Output path must not overwrite source audio.")
        if destination.suffix.lower() != ".wav":
            raise AudioPreparationError("Output path must use the .wav extension.")
        if destination.exists():
            raise AudioPreparationError("Output path already exists; refusing to overwrite it.")
        if not destination.parent.is_dir():
            raise AudioPreparationError("Output directory does not exist.")

    command = [
        ffmpeg,
        "-nostdin",
        "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            shell=False,
            timeout=330,
        )
    except subprocess.TimeoutExpired as exc:
        if temporary_directory is not None:
            temporary_directory.cleanup()
        elif destination.exists():
            destination.unlink(missing_ok=True)
        raise AudioPreparationError("FFmpeg conversion timed out.") from exc
    except subprocess.CalledProcessError as exc:
        if temporary_directory is not None:
            temporary_directory.cleanup()
        elif destination.exists():
            destination.unlink(missing_ok=True)
        raise AudioPreparationError("FFmpeg could not convert the input audio.") from exc

    if not destination.is_file() or destination.stat().st_size == 0:
        if temporary_directory is not None:
            temporary_directory.cleanup()
        elif destination.exists():
            destination.unlink(missing_ok=True)
        raise AudioPreparationError("FFmpeg did not create a valid output file.")

    try:
        validate_prepared_wav(destination)
    except AudioPreparationError:
        if temporary_directory is not None:
            temporary_directory.cleanup()
        elif destination.exists():
            destination.unlink(missing_ok=True)
        raise

    return PreparedAudio(destination, duration, temporary_directory)
