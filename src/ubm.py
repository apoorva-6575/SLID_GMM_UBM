"""
Universal Background Model (UBM) Module.

This module handles the training and saving/loading of the UBM,
which is a large Gaussian Mixture Model (GMM) trained on pooled features
from all available languages and speakers.
"""

from pathlib import Path

import joblib
from sklearn.mixture import GaussianMixture


class UBM:
    """
    Universal Background Model wrapper using scikit-learn's GaussianMixture.
    """

    def __init__(self, n_components=64, covariance_type='diag', max_iter=100, random_state=42):
        """
        Initialize the UBM.

        Parameters
        ----------
        n_components : int
            Number of Gaussian components (K).
        covariance_type : str
            Type of covariance parameters ('diag', 'full', 'tied', 'spherical').
            'diag' is standard for speech processing as it assumes features are decorrelated (which DCT does).
        max_iter : int
            Maximum number of EM iterations.
        random_state : int
            Random seed for reproducibility.
        """
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state,
            verbose=2,          # Print progress during training
            verbose_interval=10
        )
        self.is_trained = False

    def train(self, features):
        """
        Train the UBM using the Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        features : np.ndarray
            Matrix of concatenated MFCC features from all training files.
            Shape: (total_num_frames, num_features)
        """
        if len(features.shape) != 2:
            raise ValueError(f"Features must be a 2D array. Got shape {features.shape}")

        print(f"Training UBM on {features.shape[0]} frames with {self.gmm.n_components} components...")
        
        # The fit() method runs the EM algorithm until convergence or max_iter
        self.gmm.fit(features)
        
        self.is_trained = True
        print("UBM training complete.")

    def save(self, filepath):
        """Save the trained UBM to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained UBM.")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.gmm, path)
        print(f"UBM saved to {path}")

    @classmethod
    def load(cls, filepath):
        """Load a trained UBM from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        instance = cls()
        instance.gmm = joblib.load(path)
        instance.is_trained = True
        return instance
