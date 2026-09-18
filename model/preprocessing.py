"""English text tokenization and post-padding utilities."""

from __future__ import annotations

from collections.abc import Sequence

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from model.config import DEFAULT_CONFIG, ModelConfig


def fit_tokenizer(
    train_texts: Sequence[str], config: ModelConfig = DEFAULT_CONFIG
) -> Tokenizer:
    """Fit a tokenizer on training text only."""

    tokenizer = Tokenizer(
        num_words=config.max_vocabulary_size,
        oov_token=config.oov_token,
    )
    tokenizer.fit_on_texts([str(text) for text in train_texts])
    return tokenizer


def transform_texts(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    config: ModelConfig = DEFAULT_CONFIG,
):
    sequences = tokenizer.texts_to_sequences([str(text) for text in texts])
    return pad_sequences(
        sequences,
        maxlen=config.max_sequence_length,
        padding="post",
        truncating="post",
    )
