# Text classifier modules

These modules document and implement the public V10 methodology without private
data or fitted artifacts.

The classifier expects English analysis units. Thai STT output must pass through
a Thai-to-English translation interface before classification; that integration
is not implemented in this release. Compatible English input may bypass it.

Architecture: frozen 300-dimensional `glove-wiki-gigaword-300` embeddings,
parallel Conv1D branches with 64 filters and kernels 3/4/5, global max pooling,
concatenation, Dense 64, dropout 0.5, and a sigmoid output. Sequences are
post-padded/post-truncated to 100 tokens. The locked decision threshold is 0.14.

Callers must supply their own dataset paths, GloVe download/cache, trained model,
and fitted tokenizer. No Test loader exists. `training.py` accepts only explicit
Train and Validation frames. `cross_validation.py` returns fold-level aggregates
and does not export OOF rows or original text. `inference.py` loads only paths
explicitly provided by the user and classifies utterances independently.

Sigmoid scores are not calibrated probabilities. A label is a risk indicator,
not proof of fraud, and no module identifies which speaker is a scammer.

Reproducibility can vary with Python, TensorFlow, hardware, kernels, and runtime
configuration even when a seed is fixed. See `../docs/reproducibility.md`.
