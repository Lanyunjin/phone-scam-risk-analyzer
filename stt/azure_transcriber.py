"""Azure ConversationTranscriber wrapper with neutral speaker handling."""

from __future__ import annotations

import threading
from pathlib import Path

from stt.config import SpeechSettings
from stt.schemas import TranscriptionResult, TranscriptionSegment

TICKS_PER_SECOND = 10_000_000.0


class TranscriptionError(RuntimeError):
    """Raised for safe, non-credential transcription failures."""


def _sdk_time_to_seconds(value: object | None) -> float | None:
    """Convert an SDK tick count or duration-like object without assuming a type."""

    if value is None:
        return None
    total_seconds = getattr(value, "total_seconds", None)
    if callable(total_seconds):
        return float(total_seconds())
    try:
        return float(value) / TICKS_PER_SECOND
    except (TypeError, ValueError):
        return None


def transcribe_wav(
    wav_path: str | Path,
    settings: SpeechSettings | None = None,
) -> TranscriptionResult:
    """Transcribe a prepared WAV file into structured utterance segments.

    Speaker IDs are retained only as neutral diarization labels. They do not
    identify a scammer or establish speaker intent.
    """

    path = Path(wav_path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".wav":
        raise TranscriptionError("A prepared local WAV file is required.")
    settings = settings or SpeechSettings.from_environment()

    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as exc:
        raise TranscriptionError(
            "Azure Speech SDK is not installed. Install the declared dependencies."
        ) from exc

    speech_config = speechsdk.SpeechConfig(
        subscription=settings.subscription_key,
        region=settings.region,
    )
    speech_config.speech_recognition_language = settings.recognition_language
    audio_config = speechsdk.audio.AudioConfig(filename=str(path))
    transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    segments: list[TranscriptionSegment] = []
    finished = threading.Event()
    state = {"status": "running", "cancellation_reason": None}

    def on_transcribed(event: object) -> None:
        result = event.result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            segments.append(
                TranscriptionSegment(
                    segment_id=len(segments) + 1,
                    speaker_id=getattr(result, "speaker_id", None) or None,
                    text=result.text or "",
                    recognition_status="recognized",
                    offset_seconds=_sdk_time_to_seconds(
                        getattr(result, "offset", None)
                    ),
                    duration_seconds=_sdk_time_to_seconds(
                        getattr(result, "duration", None)
                    ),
                )
            )
        elif result.reason == speechsdk.ResultReason.NoMatch:
            segments.append(
                TranscriptionSegment(
                    segment_id=len(segments) + 1,
                    speaker_id=None,
                    text="",
                    recognition_status="no_match",
                )
            )

    def on_canceled(event: object) -> None:
        state["status"] = "canceled"
        reason = getattr(event, "reason", None)
        state["cancellation_reason"] = str(reason) if reason is not None else "unknown"
        finished.set()

    def on_stopped(_: object) -> None:
        if state["status"] == "running":
            state["status"] = "completed"
        finished.set()

    transcriber.transcribed.connect(on_transcribed)
    transcriber.canceled.connect(on_canceled)
    transcriber.session_stopped.connect(on_stopped)

    started = False
    try:
        transcriber.start_transcribing_async().get()
        started = True
        if not finished.wait(timeout=settings.timeout_seconds):
            state["status"] = "timed_out"
            state["cancellation_reason"] = "bounded_timeout"
    except Exception as exc:
        # Do not include SDK exception text: it can contain service details.
        raise TranscriptionError("Azure transcription failed safely.") from exc
    finally:
        if started:
            try:
                transcriber.stop_transcribing_async().get()
            except Exception:
                if state["status"] not in {"canceled", "timed_out"}:
                    state["status"] = "stop_failed"

    return TranscriptionResult(
        recognition_language=settings.recognition_language,
        status=str(state["status"]),
        segments=segments,
        cancellation_reason=state["cancellation_reason"],
    )
