# Label definition

The classification unit is one English text analysis unit. Units are evaluated
independently rather than as complete calls.

## Label 0 — Safe

The unit does not contain sufficient standalone textual evidence of scam risk.
This label does not guarantee that a broader interaction is safe.

## Label 1 — Risky

The unit contains standalone textual patterns that may indicate scam risk, such
as coercive urgency, requests for sensitive information, or instructions to
bypass normal safeguards. This label is a risk indication, not proof of fraud.

The classifier does not infer speaker identity, intent, or which speaker is a
scammer. Speaker diarization labels are not classification labels.
