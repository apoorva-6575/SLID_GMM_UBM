"""Language-specific GMM helpers."""

import joblib
from sklearn.mixture import GaussianMixture


def train_gmm(features, n_components=32, covariance_type="diag", random_state=42):
    """Train a GMM for one language class."""
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=100,
        random_state=random_state,
    )
    model.fit(features)
    return model


def score_gmm(model, features):
    """Return mean log-likelihood for a feature matrix."""
    return model.score(features)


def save_gmm(model, path):
    """Persist a trained GMM."""
    joblib.dump(model, path)
