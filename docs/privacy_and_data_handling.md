# Privacy and data handling

This public repository excludes all real V9/V10 rows, full master datasets,
Test files, row-level predictions, OOF outputs, FP/FN rows, real audio, trained
weights, fitted tokenizers, checkpoints, full GloVe files, and private hashes.

Synthetic files are newly authored schema demonstrations. They contain no real
person, organization, account, phone number, transaction identifier, or copied
conversation. They must not be treated as evidence about the private dataset.

For any future audio processing, obtain appropriate consent, minimize retention,
restrict access, and define deletion procedures before collection. Do not place
real audio or transcripts in this repository. Review the Azure service's current
privacy and data-handling terms before use.

Credentials are supplied only through `AZURE_SPEECH_KEY` and
`AZURE_SPEECH_REGION`. Code must not print these values or raw service errors
that might expose private configuration. The old prototype credential is not
valid for reuse and must remain revoked/rotated.
