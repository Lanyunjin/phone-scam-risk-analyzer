"""Stable, JSON-serializable speech transcription result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TranscriptionSegment:
    segment_id: int
    speaker_id: str | None
    text: str
    recognition_status: str
    offset_seconds: float | None = None
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TranscriptionResult:
    recognition_language: str
    status: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    cancellation_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recognition_language": self.recognition_language,
            "status": self.status,
            "segments": [segment.to_dict() for segment in self.segments],
            "cancellation_reason": self.cancellation_reason,
        }
