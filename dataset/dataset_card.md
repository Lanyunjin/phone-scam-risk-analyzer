# Dataset card: private V10 design

## Availability

The real V10 dataset is private and is not distributed in this repository.
Redistribution rights remain under review. This document describes only the
approved high-level design and aggregate split counts.

## Intended task

Binary classification of an English text analysis unit:

- `0`: Safe
- `1`: Risky

Each analysis unit is classified independently. A label represents textual risk
evidence, not proof of fraud and not the identity or intent of any speaker.

## Splits

- Train: 1,164 rows — Safe 605 / Risky 559
- Validation: 197 rows — Safe 100 / Risky 97
- Test: 200 rows — Safe 100 / Risky 100

Test remained isolated from model development. It was not used for training,
architecture selection, or threshold selection.

## Public sample

`sample_dataset.csv` contains only newly written fictional statements with
`SAMPLE_TYPE=synthetic_demonstration`. It contains no real personal data,
names, phone numbers, organizations, accounts, transaction identifiers, or
conversation excerpts. No public sample comes from Train, Validation, or Test.
Real data must not be inferred from the synthetic examples.

## Limitations

The aggregate counts do not describe source distribution, demographic coverage,
linguistic variation, collection conditions, or real-world prevalence. The
private dataset should not be assumed representative of all scam scenarios.
Public synthetic rows demonstrate only file shape and label semantics.
