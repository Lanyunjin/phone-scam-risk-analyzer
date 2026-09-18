# Phone Scam Risk Analyzer

Phone Scam Risk Analyzer is an academic prototype for identifying scam-risk
evidence in individual text analysis units derived from phone-call audio. It was
developed as a university computer science project. This repository is public
for academic review and portfolio demonstration; it is not a production fraud
detection service and does not contain a trained production model.

## Scope

This release presents three areas only:

1. Dataset design and data-preparation methodology.
2. Development methodology for an English multi-kernel TextCNN classifier.
3. A public-safe Azure Speech-to-Text pipeline for Thai audio.

The backend integration and translation stage are still in development. The
mobile application is not included.

## System pipeline and language boundary

```text
Thai audio
  -> FFmpeg conversion to 16 kHz mono PCM signed 16-bit WAV
  -> Azure Speech transcription with language th-TH
  -> utterance-level Thai text with neutral speaker labels
  -> Thai-to-English translation boundary (implementation not finalized)
  -> English analysis unit
  -> frozen-GloVe multi-kernel CNN
  -> Safe (0) or Risky (1) risk indication
```

English text that already matches the classifier's documented preprocessing may
bypass translation. No translation provider is selected or implemented here.
Every utterance is classified independently. Speaker diarization identifiers
are neutral labels and do not identify which speaker is fraudulent.

## Repository contents

- `dataset/`: design notes, label definitions, and synthetic demonstration rows.
- `stt/`: local audio preparation and Azure ConversationTranscriber wrapper.
- `model/`: reusable architecture, preprocessing, training, evaluation, and
  inference utilities.
- `experiments/`: sanitized configuration and aggregate-only reporting.
- `examples/`: synthetic structured input and output.
- `docs/`: architecture, methodology, privacy, reproducibility, limitations,
  and third-party attribution.
- `tests/`: tests using only synthetic or programmatically generated data.

## Dataset design and labels

The classifier operates on one English text analysis unit at a time:

- `0 = Safe`
- `1 = Risky`

Known aggregate split composition:

| Split | Rows | Safe | Risky |
|---|---:|---:|---:|
| Train | 1,164 | 605 | 559 |
| Validation | 197 | 100 | 97 |
| Test | 200 | 100 | 100 |

The real V9 and V10 datasets are not distributed, and their redistribution
rights remain under review. Test remained isolated from model development. No
public sample is taken from Train, Validation, or Test. The CSV in `dataset/`
contains newly authored fictional demonstrations only and must not be used to
infer real data.

## Speech-to-Text

The STT module accepts WAV, MP3, or M4A input, rejects audio longer than five
minutes, and uses FFmpeg/ffprobe to prepare 16 kHz, mono, PCM signed 16-bit WAV.
Azure Speech uses `ConversationTranscriber` with `th-TH` by default and returns
structured utterance segments. It handles recognized speech, no-match events,
cancellation, completion, and a bounded timeout without logging credentials.

FFmpeg and ffprobe are external system dependencies and must be available on
`PATH`. They are not installed by `pip`.

## Classifier architecture

The public architecture is a multi-kernel TextCNN:

- Frozen `glove-wiki-gigaword-300` embedding layer, dimension 300.
- Maximum sequence length 100, post-padding, and post-truncation.
- Three Conv1D branches with 64 filters and kernel sizes 3, 4, and 5.
- Global max pooling for each branch, followed by concatenation.
- Dense layer with 64 units, dropout 0.5, and sigmoid binary output.
- Locked classification threshold: 0.14.

Sigmoid scores are not calibrated probabilities. A `Risky` output is a model
risk indication and does not prove fraud.

The repository does not ship trained weights, a fitted tokenizer, checkpoints,
or GloVe vectors. GloVe is downloaded separately through Gensim after reviewing
the applicable third-party terms.

## Training, validation, and evaluation

Development used grouped, stratified cross-validation so related groups did not
cross fold boundaries. Tokenizers and embedding matrices were fit or constructed
inside each training fold. Public utilities report aggregate metrics and do not
write original text, OOF predictions, or row-level error reports by default.

The final classification threshold was selected before held-out Test evaluation.
Test was not used for training, architecture selection, or threshold selection.
Approved aggregate results for the text classifier are:

| Rows | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.14 | 0.9400 | 0.9314 | 0.9500 | 0.9406 | 0.9871 |

These metrics apply only to the text classifier. They are not performance
metrics for the complete audio, transcription, translation, backend, or mobile
pipeline. Real-world performance may differ.

## Installation

Python 3.11 or 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install FFmpeg separately and verify both `ffmpeg` and `ffprobe` are on `PATH`.

Configure Azure locally without committing credentials:

```powershell
Copy-Item .env.example .env
$env:AZURE_SPEECH_KEY = "your-own-key"
$env:AZURE_SPEECH_REGION = "eastasia"
```

The code reads environment variables directly; it does not automatically parse
`.env`. Never commit the real `.env` file.

## Safe local checks and examples

No Azure call, model training, GloVe download, or private artifact is required
for the included unit tests:

```powershell
python -m pytest tests
python -m json.tool examples/synthetic_transcript.json
python -m json.tool experiments/reports/v10_final_test_aggregate.json
```

The JSON and CSV demonstrations are synthetic and were not used for training or
evaluation.

## Privacy and limitations

- No real datasets, real audio, Test rows, row-level predictions, trained model,
  fitted tokenizer, or full GloVe files are included.
- Transcription and translation errors can propagate into classification.
- Translation integration is not complete.
- Utterances are classified without broader conversational context.
- Speaker IDs do not identify a scammer.
- Predictions are not legal, financial, or safety determinations.
- Performance may shift across speakers, domains, accents, devices, noise, and
  language varieties.

See `docs/privacy_and_data_handling.md` and `docs/limitations.md`.

## Current project status

- Dataset V10 preparation: Completed
- Train/Validation/Test split: Completed
- V10 model development: Completed
- Final held-out evaluation: Completed privately
- Test data and row-level Test outputs: Private
- Backend/inference integration: In development
- Translation integration: In development
- Mobile application: In development and excluded from this release

Project authors are intentionally omitted from this first public release.

## Rights and acknowledgements

Copyright © 2026 Phone Scam Risk Analyzer Project Authors. All rights reserved.
This is not open-source software. See `NOTICE` for the permitted purpose and
restrictions.

The project references Azure Cognitive Services Speech SDK, TensorFlow/Keras,
Gensim, GloVe, scikit-learn, NumPy, and pandas. These components retain their
separate licenses and terms; see `docs/third_party_attributions.md`.
