"""Reusable training entry point for caller-supplied development data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf

from model.architecture import build_multikernel_textcnn
from model.config import DEFAULT_CONFIG, ModelConfig
from model.data import LABEL_COLUMN, TEXT_COLUMN, validate_labeled_frame
from model.embeddings import build_embedding_matrix
from model.preprocessing import fit_tokenizer, transform_texts


@dataclass
class TrainingResult:
    model: tf.keras.Model
    tokenizer: object
    history: dict[str, list[float]]
    embedding_summary: dict[str, float | int]


def train_with_validation(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    glove_vectors,
    config: ModelConfig = DEFAULT_CONFIG,
) -> TrainingResult:
    """Train from explicit Train/Validation frames; no Test data is accepted or loaded."""

    train = validate_labeled_frame(train_frame)
    validation = validate_labeled_frame(validation_frame)
    tokenizer = fit_tokenizer(train[TEXT_COLUMN].tolist(), config)
    x_train = transform_texts(tokenizer, train[TEXT_COLUMN].tolist(), config)
    x_validation = transform_texts(tokenizer, validation[TEXT_COLUMN].tolist(), config)
    embedding_matrix, summary = build_embedding_matrix(tokenizer, glove_vectors, config)

    tf.keras.utils.set_random_seed(config.random_seed)
    model = build_multikernel_textcnn(embedding_matrix, config)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.early_stopping_patience,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        train[LABEL_COLUMN].to_numpy(dtype=np.int32),
        validation_data=(
            x_validation,
            validation[LABEL_COLUMN].to_numpy(dtype=np.int32),
        ),
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )
    return TrainingResult(model, tokenizer, history.history, summary)
