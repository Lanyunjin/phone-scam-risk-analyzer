"""Approved frozen-GloVe multi-kernel TextCNN architecture."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from model.config import DEFAULT_CONFIG, ModelConfig


def build_multikernel_textcnn(
    embedding_matrix: np.ndarray,
    config: ModelConfig = DEFAULT_CONFIG,
) -> tf.keras.Model:
    if embedding_matrix.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional.")
    if embedding_matrix.shape[1] != config.embedding_dimension:
        raise ValueError("Embedding matrix dimension does not match configuration.")

    inputs = tf.keras.Input(shape=(config.max_sequence_length,), name="token_ids")
    embedded = tf.keras.layers.Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=config.embedding_dimension,
        weights=[embedding_matrix],
        trainable=False,
        name="frozen_glove_embedding",
    )(inputs)

    branches = []
    for filters, kernel_size in zip(
        config.convolution_filters,
        config.kernel_sizes,
        strict=True,
    ):
        convolved = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            name=f"conv_kernel_{kernel_size}",
        )(embedded)
        branches.append(
            tf.keras.layers.GlobalMaxPooling1D(
                name=f"global_max_pool_kernel_{kernel_size}"
            )(convolved)
        )

    combined = tf.keras.layers.Concatenate(name="concatenate_pooled_features")(
        branches
    )
    hidden = tf.keras.layers.Dense(
        config.dense_units,
        activation="relu",
        name="dense_features",
    )(combined)
    dropped = tf.keras.layers.Dropout(config.dropout_rate, name="dropout")(hidden)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="risk_score")(
        dropped
    )
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="multikernel_textcnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
    )
    return model
