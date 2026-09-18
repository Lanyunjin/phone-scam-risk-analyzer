"""Azure Speech configuration loaded exclusively from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class SpeechConfigurationError(RuntimeError):
    """Raised when required non-secret configuration is unavailable."""


@dataclass(frozen=True)
class SpeechSettings:
    """Runtime settings. The credential value must never be logged or serialized."""

    subscription_key: str = field(repr=False)
    region: str
    recognition_language: str = "th-TH"
    timeout_seconds: float = 330.0

    @classmethod
    def from_environment(cls) -> "SpeechSettings":
        key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
        region = os.environ.get("AZURE_SPEECH_REGION", "").strip()
        if not key:
            raise SpeechConfigurationError(
                "Required environment variable AZURE_SPEECH_KEY is not set."
            )
        if not region:
            raise SpeechConfigurationError(
                "Required environment variable AZURE_SPEECH_REGION is not set."
            )
        return cls(subscription_key=key, region=region)
