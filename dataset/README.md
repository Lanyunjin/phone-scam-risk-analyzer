# Dataset documentation

This directory contains documentation and newly authored synthetic
demonstrations only. It does not contain Dataset V9, Dataset V10, or a public
training/evaluation dataset.

## Aggregate split composition

| Split | Rows | Safe (0) | Risky (1) |
|---|---:|---:|---:|
| Train | 1,164 | 605 | 559 |
| Validation | 197 | 100 | 97 |
| Test | 200 | 100 | 100 |

Each text analysis unit is classified independently. Test remained isolated
from training, architecture selection, and classification-threshold selection.
The real datasets are not distributed, and dataset redistribution rights remain
under review.

`sample_dataset.csv` is a schema demonstration. Every row was written from
scratch for this public repository. No row was copied, adapted, paraphrased, or
derived from Train, Validation, Test, or a real conversation. The examples do
not reveal or approximate real project data and are not suitable for training
or evaluation.
