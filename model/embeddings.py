"""GloVe download/loading and frozen embedding-matrix construction."""

from __future__ import annotations

import numpy as np

from model.config import DEFAULT_CONFIG, ModelConfig


def load_glove(config: ModelConfig = DEFAULT_CONFIG):
    """Download/load GloVe through Gensim under its separate terms."""

    import gensim.downloader as api

    vectors = api.load(config.embedding_name)
    if vectors.vector_size != config.embedding_dimension:
        raise ValueError("Loaded embeddings have an unexpected dimension.")
    return vectors


def build_embedding_matrix(tokenizer, vectors, config: ModelConfig = DEFAULT_CONFIG):
    input_dimension = min(
        config.max_vocabulary_size,
        len(tokenizer.word_index) + 1,
    )
    matrix = np.zeros(
        (input_dimension, config.embedding_dimension),
        dtype=np.float32,
    )
    matched = 0
    for word, index in tokenizer.word_index.items():
        if index >= input_dimension:
            continue
        if word == config.oov_token.lower() or word not in vectors:
            continue
        matrix[index] = vectors[word]
        matched += 1
    vocabulary_size = max(input_dimension - 1, 0)
    return matrix, {
        "vocabulary_size": vocabulary_size,
        "matched_embeddings": matched,
        "embedding_coverage": matched / vocabulary_size if vocabulary_size else 0.0,
    }
