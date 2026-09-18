"""Inference helpers for user-supplied local model and tokenizer artifacts."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from model.config import DEFAULT_CONFIG, ModelConfig
from model.preprocessing import transform_texts


def load_user_artifacts(model_path: str | Path, tokenizer_path: str | Path):
    model_file = Path(model_path).expanduser()
    tokenizer_file = Path(tokenizer_path).expanduser()
    if not model_file.is_file() or not tokenizer_file.is_file():
        raise FileNotFoundError("Both user-supplied artifact paths must exist.")
    tokenizer_payload = tokenizer_file.read_text(encoding="utf-8")
    tokenizer = tokenizer_from_json(tokenizer_payload)
    model = tf.keras.models.load_model(model_file)
    return model, tokenizer


def classify_utterances(
    texts: list[str],
    model,
    tokenizer,
    config: ModelConfig = DEFAULT_CONFIG,
) -> list[dict[str, object]]:
    """Classify each English utterance independently.

    Returned sigmoid scores are not calibrated probabilities. Results do not
    identify a speaker or prove fraud.
    """

    if not texts or any(not str(text).strip() for text in texts):
        raise ValueError("At least one non-empty English analysis unit is required.")
    inputs = transform_texts(tokenizer, texts, config)
    scores = model.predict(inputs, batch_size=config.batch_size, verbose=0).reshape(-1)
    return [
        {
            "analysis_unit_id": index + 1,
            "label": int(score >= config.threshold),
            "label_name": "Risky" if score >= config.threshold else "Safe",
            "score": float(score),
            "threshold": config.threshold,
            "score_is_calibrated_probability": False,
        }
        for index, score in enumerate(scores)
    ]
