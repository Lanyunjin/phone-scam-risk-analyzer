"""Aggregate-only binary classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def aggregate_binary_metrics(y_true, scores, threshold: float) -> dict[str, float | int | None]:
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    values = np.asarray(scores, dtype=float).reshape(-1)
    if labels.size == 0 or labels.size != values.size:
        raise ValueError("Labels and scores must be non-empty and have equal length.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must be binary.")
    predictions = (values >= threshold).astype(int)
    auc = float(roc_auc_score(labels, values)) if np.unique(labels).size == 2 else None
    return {
        "rows": int(labels.size),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": auc,
    }
