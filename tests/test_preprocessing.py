from __future__ import annotations

import pandas as pd
import pytest

from model.data import DatasetSchemaError, validate_labeled_frame


def test_synthetic_schema_accepts_binary_labels():
    frame = pd.DataFrame(
        {
            "CLEANED_TEXT": ["a fictional safe unit", "a fictional risky unit"],
            "LABEL_BIN": [0, 1],
        }
    )
    validated = validate_labeled_frame(frame)
    assert validated["LABEL_BIN"].tolist() == [0, 1]


def test_schema_rejects_nonbinary_label():
    frame = pd.DataFrame({"CLEANED_TEXT": ["synthetic"], "LABEL_BIN": [2]})
    with pytest.raises(DatasetSchemaError, match=r"0 \(Safe\) or 1 \(Risky\)"):
        validate_labeled_frame(frame)


def test_post_padding_and_truncation():
    pytest.importorskip("tensorflow")
    from model.config import ModelConfig
    from model.preprocessing import fit_tokenizer, transform_texts

    config = ModelConfig(max_sequence_length=4)
    tokenizer = fit_tokenizer(["one two three four five"], config)
    values = transform_texts(tokenizer, ["one two"], config)
    assert values.shape == (1, 4)
    assert values[0, 2:].tolist() == [0, 0]
