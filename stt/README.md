# Speech-to-Text module

This module prepares local WAV, MP3, or M4A input and transcribes it with Azure
Cognitive Services Speech SDK `ConversationTranscriber`.

## Requirements

- Maximum source duration: 300 seconds.
- Output format: WAV, 16 kHz, mono, PCM signed 16-bit.
- Recognition language: `th-TH` by default.
- FFmpeg and ffprobe must be installed externally and available on `PATH`.
- `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` must be set in the environment.

The source file is never overwritten. If no output path is provided,
`prepare_audio` uses a temporary directory that can be cleaned with its context
manager.

```python
from stt.audio_preprocessing import prepare_audio
from stt.azure_transcriber import transcribe_wav

with prepare_audio("local_input.m4a") as prepared:
    result = transcribe_wav(prepared.path)
    payload = result.to_dict()
```

Do not use real audio without appropriate consent and a retention/deletion plan.
The returned `speaker_id` is a neutral diarization label. It must never be
described as identifying the scammer.

Thai output reaches a translation interface boundary before the English GloVe
classifier. Translation integration is not implemented or assigned to a
provider in this release.

The wrapper returns structured data and does not print credentials. Cancellation
details are deliberately reduced to safe status information rather than raw
service error text.
