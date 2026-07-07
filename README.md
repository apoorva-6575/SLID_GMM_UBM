# SLID_GMM_UBM

Spoken Language Identification using GMM-UBM, language-specific GMMs, and ANN baselines.

## Project Structure

```text
SLID_GMM_UBM/
├── data/
│   ├── train/
│   │   ├── audio/
│   │   └── metadata.csv
│   ├── valid/
│   │   ├── audio/
│   │   └── metadata.csv
│   ├── test/
│   │   ├── audio/
│   │   └── metadata.csv
│   └── README.md
├── notebooks/
│   └── Dataset_Preparation.ipynb
├── src/
│   ├── audio_loader.py
│   ├── preprocessing.py
│   ├── framing.py
│   ├── mfcc.py
│   ├── feature_extraction.py
│   ├── ubm.py
│   ├── gmm.py
│   ├── ann.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/
│   └── gmm_models/
├── results/
│   ├── plots/
│   └── logs/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Data

Keep raw audio files outside Git. Place them locally under:

```text
data/train/audio/
data/valid/audio/
data/test/audio/
```

Small metadata CSV files can be tracked:

```text
data/train/metadata.csv
data/valid/metadata.csv
data/test/metadata.csv
```

The old Hugging Face download/cache folders `dataset/` and `hf_cache/` are ignored.

## Setup

```bash
pip install -r requirements.txt
```

## Notes

- `features/`, `models/`, and `results/` are local output folders.
- Model files such as `.pkl`, `.pt`, and `.pth` are ignored by Git.
- Kaggle credentials such as `kaggle.json` must never be committed.
