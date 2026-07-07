"""Universal Background Model helpers."""

import joblib
from sklearn.mixture import GaussianMixture


def train_ubm(features, n_components=64, covariance_type="diag", random_state=42):
    """Train a GMM-based UBM from pooled features."""
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=100,
        random_state=random_state,
        verbose=1,
    )
    model.fit(features)
    return model


def save_ubm(model, path):
    """Persist a trained UBM."""
    joblib.dump(model, path)
