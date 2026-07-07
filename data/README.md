# Data

Use this folder for local train, validation, and test splits.

Expected layout:

```text
data/
├── train/
│   ├── audio/
│   └── metadata.csv
├── valid/
│   ├── audio/
│   └── metadata.csv
└── test/
    ├── audio/
    └── metadata.csv
```

Audio files are ignored by Git. Metadata CSV files are small enough to commit when they do not contain sensitive information.
