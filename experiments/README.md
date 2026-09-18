# Sanitized experiment materials

This directory contains a public architecture/configuration example and one
approved aggregate-only held-out Test report. It contains no dataset, trained
artifact, fitted tokenizer, prediction, row ID, original text, per-row score,
FP/FN example, private path, or fingerprint.

The threshold was selected before the final held-out Test evaluation. Test was
not used for training, architecture selection, or threshold selection. The
reported metrics apply only to the English text classifier and do not measure
the complete audio-to-application pipeline.

Use the modules under `model/` with caller-supplied development data. Test
evaluation is intentionally outside the public training workflow.
