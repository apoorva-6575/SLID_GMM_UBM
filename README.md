# SLID_GMM_UBM

Spoken Language Identification for selected IndicVoices languages using MFCC features, a GMM-UBM framework, and an ANN classifier.

## Objective

This project is for language identification, not speaker identification.

- Input: speech audio from selected IndicVoices languages
- Features: MFCCs and optional delta features
- Modeling: GMM-UBM framework
- Classifier: ANN
- Output: predicted language, such as Hindi, Bengali, Nepali, Punjabi, or Urdu

Speaker IDs are used only to create speaker-independent train, validation, and test splits. This prevents the model from learning speaker-specific traits instead of language characteristics, making evaluation more meaningful.

## Project Structure

```text
SLID_GMM_UBM/
├── src/
│   ├── preprocessing.py          # Audio loading, normalization, resampling
│   ├── framing.py                # Framing and windowing
│   ├── mfcc.py                   # MFCC and delta extraction
│   ├── feature_extraction.py     # End-to-end feature pipeline
│   ├── gmm.py                    # Language-specific GMM
│   ├── ubm.py                    # Universal Background Model
│   ├── ann.py                    # ANN classifier (SLIDNet)
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation metrics
│   └── utils.py                  # General utilities
│
├── notebooks/
│   ├── dataset_exploration.ipynb
│   └── dataset_preparation.ipynb
│
├── configs/                      # Hyperparameters and experiment configs
│
├── data/                         # Metadata CSVs (audio provided via Kaggle)
│   ├── train/
│   │   ├── audio/
│   │   └── metadata.csv
│   ├── valid/
│   │   ├── audio/
│   │   └── metadata.csv
│   └── test/
│       ├── audio/
│       └── metadata.csv
│
├── models/
│   └── gmm_models/
│
├── results/
│   ├── metrics/
│   ├── plots/
│   ├── embeddings/
│   └── confusion_matrix/
│
├── docs/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Data

The IndicVoices dataset is hosted on Kaggle. Audio files are never committed to this repository.

Place audio files locally under:

```text
data/train/audio/
data/valid/audio/
data/test/audio/
```

Small metadata CSV files are tracked in Git:

```text
data/train/metadata.csv
data/valid/metadata.csv
data/test/metadata.csv
```

## Setup

```bash
pip install -r requirements.txt
```

## Notes

- `models/` and `results/` are local output folders. Their contents are ignored by Git.
- Audio files (`.wav`, `.mp3`, `.flac`) are ignored by Git.
- Model files (`.pkl`, `.pt`, `.pth`) are ignored by Git.
- Kaggle credentials (`kaggle.json`) must never be committed.
- The dataset is attached separately through Kaggle, not stored in this repository.

## License

MIT License. See [LICENSE](LICENSE) for details.
