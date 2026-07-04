# SLID_GMM_UBM

Spoken language identification project using GMM-UBM style experiments.

## Project Structure

- `src/` - source scripts
- `notebooks/` - experiment notebooks
- `dataset/` - local dataset cache, ignored by Git
- `features/` - extracted feature files, ignored by Git
- `models/` - trained model files, ignored by Git
- `results/` - generated outputs, ignored by Git

## Dataset

The dataset can be downloaded with:

```bash
python src/download_dataset.py
```

Large data, feature, model, and result files are intentionally not committed to the repository.
