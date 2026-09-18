# Methodology

## Data roles

The completed V10 design used distinct Train, Validation, and held-out Test
roles. Approved aggregate counts are:

- Train: 1,164 — Safe 605 / Risky 559
- Validation: 197 — Safe 100 / Risky 97
- Test: 200 — Safe 100 / Risky 100

Test remained isolated from model development. It was not used for training,
architecture selection, or threshold selection. Real rows are private and are
not represented by the synthetic public examples.

## Development

Grouped stratified cross-validation was used conceptually to preserve class
balance while keeping related groups within a single fold. The tokenizer was fit
on fold-training text only, and the fold embedding matrix was constructed after
that fit. Model selection used aggregate development evidence rather than the
held-out Test split.

The final candidate is a frozen-GloVe multi-kernel TextCNN. Validation was used
for the pre-Test selection of the locked 0.14 classification threshold.

## Held-out evaluation

The final held-out evaluation was completed privately. Approved aggregate
text-classifier results are recorded in
`../experiments/reports/v10_final_test_aggregate.json`. The report contains no
predictions, sample identifiers, row numbers, original text, per-row scores,
confusion counts, error examples, private paths, or fingerprints.

The approved figures apply only to the English text classifier. They do not
measure complete audio capture, Thai STT, translation, backend integration, or
mobile use. Real-world performance may differ.
