# Limitations

- The classifier evaluates each English analysis unit independently and lacks
  full-call context.
- Thai speech requires transcription and then translation. Translation
  integration is still in development and can introduce meaning changes.
- Speech recognition can be affected by noise, overlap, accents, microphones,
  compression, and connectivity.
- Speaker diarization IDs are neutral labels and do not identify a scammer.
- A Risky classification is not proof of fraud; a Safe classification is not a
  guarantee of safety.
- Sigmoid scores are not calibrated probabilities.
- The approved 0.9400 accuracy is a private held-out text-classifier result, not
  guaranteed real-world accuracy and not an end-to-end audio result.
- Dataset redistribution rights remain under review, and coverage should not be
  assumed representative of every population, language variety, or scam type.
- Backend, translation, and mobile integration remain incomplete.
