"""Public, dataset-independent model configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    max_vocabulary_size: int = 10_000
    max_sequence_length: int = 100
    oov_token: str = "<OOV>"
    embedding_name: str = "glove-wiki-gigaword-300"
    embedding_dimension: int = 300
    convolution_filters: tuple[int, int, int] = (64, 64, 64)
    kernel_sizes: tuple[int, int, int] = (3, 4, 5)
    dense_units: int = 64
    dropout_rate: float = 0.5
    threshold: float = 0.14
    batch_size: int = 32
    max_epochs: int = 50
    learning_rate: float = 0.001
    early_stopping_patience: int = 3
    random_seed: int = 42
    cross_validation_splits: int = 5

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["convolution_filters"] = list(self.convolution_filters)
        payload["kernel_sizes"] = list(self.kernel_sizes)
        return payload


DEFAULT_CONFIG = ModelConfig()
