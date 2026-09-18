# Reproducibility

Python 3.11 or 3.12 is recommended. Dependency ranges appear in
`requirements.txt`; exact results can vary across TensorFlow versions, CPU/GPU
hardware, kernels, operating systems, and nondeterministic runtime operations.

The public configuration records seed 42, five grouped folds, maximum sequence
length 100, training limits, architecture parameters, and threshold 0.14.
Fold-local tokenizer fitting and embedding construction are required to avoid
development leakage.

The repository cannot reproduce the private reported metrics by itself because
it intentionally omits real datasets, fitted tokenizers, trained weights, and
full GloVe files. The public code demonstrates the methodology with caller-owned
data and artifacts. It must not be pointed automatically at Test data.

GloVe may be downloaded through Gensim after separately reviewing its terms.
That download is large and is not required for repository unit tests.
