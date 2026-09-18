# Architecture

## End-to-end design boundary

```text
Thai audio
  -> local format and duration validation
  -> FFmpeg: 16 kHz mono PCM signed 16-bit WAV
  -> Azure ConversationTranscriber (th-TH)
  -> Thai utterance segments and neutral speaker IDs
  -> Thai-to-English translation interface (in development)
  -> independent English analysis units
  -> frozen-GloVe multi-kernel TextCNN
  -> Safe/Risky risk indication
```

The translation provider and implementation are intentionally unspecified. The
classifier accepts English text compatible with its preprocessing. STT and
classification outputs are joined by structured application code that is still
in development.

## TextCNN

Input token sequences are post-padded or post-truncated to 100 tokens. A frozen
300-dimensional GloVe embedding feeds three parallel Conv1D branches. Each uses
64 filters with kernel size 3, 4, or 5 and global max pooling. The pooled vectors
are concatenated and passed through Dense 64, dropout 0.5, and a sigmoid output.
The locked classification threshold is 0.14.

The score is not a calibrated probability. Utterances are independent analysis
units. The model neither reasons over an entire call nor identifies a scammer
speaker.
