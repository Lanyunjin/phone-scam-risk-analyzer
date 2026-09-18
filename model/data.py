"""Schema validation for explicitly supplied development datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TEXT_COLUMN = "CLEANED_TEXT"
LABEL_COLUMN = "LABEL_BIN"
GROUP_COLUMN = "GROUP_ID"


class DatasetSchemaError(ValueError):
    """Raised when a user-supplied dataset does not match the public schema."""


def validate_labeled_frame(
    frame: pd.DataFrame, *, require_group: bool = False
) -> pd.DataFrame:
    required = {TEXT_COLUMN, LABEL_COLUMN}
    if require_group:
        required.add(GROUP_COLUMN)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DatasetSchemaError(f"Dataset is missing required columns: {missing}")

    clean = frame.loc[:, sorted(required)].copy()
    if clean[TEXT_COLUMN].isna().any() or clean[TEXT_COLUMN].astype(str).str.strip().eq("").any():
        raise DatasetSchemaError("Text values must be non-empty.")
    labels = pd.to_numeric(clean[LABEL_COLUMN], errors="raise")
    if not set(labels.unique()).issubset({0, 1}):
        raise DatasetSchemaError("Labels must contain only 0 (Safe) or 1 (Risky).")
    clean[LABEL_COLUMN] = labels.astype(int)
    clean[TEXT_COLUMN] = clean[TEXT_COLUMN].astype(str)
    if require_group:
        if clean[GROUP_COLUMN].isna().any() or clean[GROUP_COLUMN].astype(str).str.strip().eq("").any():
            raise DatasetSchemaError("Group identifiers must be non-empty.")
        clean[GROUP_COLUMN] = clean[GROUP_COLUMN].astype(str)
    return clean


def load_labeled_csv(path: str | Path, *, require_group: bool = False) -> pd.DataFrame:
    """Load only the exact path supplied by the caller; no Test path is built in."""

    supplied = Path(path).expanduser()
    if not supplied.is_file():
        raise FileNotFoundError("The supplied dataset path is not a file.")
    return validate_labeled_frame(pd.read_csv(supplied), require_group=require_group)
