"""Grouped cross-validation with aggregate-only output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from model.architecture import build_multikernel_textcnn
from model.config import DEFAULT_CONFIG, ModelConfig
from model.data import GROUP_COLUMN, LABEL_COLUMN, TEXT_COLUMN, validate_labeled_frame
from model.embeddings import build_embedding_matrix
from model.evaluation import aggregate_binary_metrics
from model.preprocessing import fit_tokenizer, transform_texts


def run_grouped_cross_validation(
    frame: pd.DataFrame,
    glove_vectors,
    config: ModelConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Return one aggregate row per fold without OOF rows or original text."""

    data = validate_labeled_frame(frame, require_group=True)
    splitter = StratifiedGroupKFold(
        n_splits=config.cross_validation_splits,
        shuffle=True,
        random_state=config.random_seed,
    )
    fold_summaries: list[dict[str, float | int | None]] = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(data[TEXT_COLUMN], data[LABEL_COLUMN], data[GROUP_COLUMN]),
        start=1,
    ):
        train = data.iloc[train_indices]
        validation = data.iloc[validation_indices]
        if set(train[GROUP_COLUMN]).intersection(validation[GROUP_COLUMN]):
            raise AssertionError("A group crosses the fold boundary.")

        tokenizer = fit_tokenizer(train[TEXT_COLUMN].tolist(), config)
        x_train = transform_texts(tokenizer, train[TEXT_COLUMN].tolist(), config)
        x_validation = transform_texts(
            tokenizer, validation[TEXT_COLUMN].tolist(), config
        )
        embedding_matrix, _ = build_embedding_matrix(tokenizer, glove_vectors, config)
        tf.keras.utils.set_random_seed(config.random_seed)
        model = build_multikernel_textcnn(embedding_matrix, config)
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        )
        model.fit(
            x_train,
            train[LABEL_COLUMN].to_numpy(dtype=np.int32),
            validation_data=(
                x_validation,
                validation[LABEL_COLUMN].to_numpy(dtype=np.int32),
            ),
            epochs=config.max_epochs,
            batch_size=config.batch_size,
            callbacks=[callback],
            verbose=0,
        )
        scores = model.predict(
            x_validation,
            batch_size=config.batch_size,
            verbose=0,
        ).reshape(-1)
        fold_summaries.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                **aggregate_binary_metrics(
                    validation[LABEL_COLUMN], scores, config.threshold
                ),
            }
        )
        tf.keras.backend.clear_session()

    return pd.DataFrame(fold_summaries)
