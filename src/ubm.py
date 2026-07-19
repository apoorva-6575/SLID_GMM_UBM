"""
Universal Background Model (UBM) Module.

This module handles the training and saving/loading of the UBM,
which is a large Gaussian Mixture Model (GMM) trained on pooled features
from all available languages and speakers.
"""

from pathlib import Path

import joblib
import numpy as np
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
            reg_covar=1e-3,     # Increased regularization to prevent singular covariance matrices
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

    def extract_supervector(self, features, relevance_factor=16.0):
        """
        Extracts a GMM-UBM supervector using MAP mean adaptation.
        
        Parameters
        ----------
        features : np.ndarray
            Matrix of MFCC features for a single utterance. Shape: (num_frames, num_features)
        relevance_factor : float
            Controls how much to trust the new data vs the UBM prior. Standard value is 16.
            
        Returns
        -------
        np.ndarray
            Flattened supervector of shape (n_components * num_features,)
        """
        if not self.is_trained:
            raise RuntimeError("Cannot extract supervector from untrained UBM.")
            
        # 1. Compute posterior probabilities (responsibilities)
        # predict_proba returns shape (num_frames, n_components)
        responsibilities = self.gmm.predict_proba(features)
        
        # 2. Compute zero-order statistics (N_k) for each component
        # sum over frames: shape (n_components,)
        n_k = np.sum(responsibilities, axis=0)
        
        # 3. Compute first-order statistics (E_k) for each component
        # shape (n_components, num_features)
        # Avoid division by zero by adding a small epsilon
        e_k = np.dot(responsibilities.T, features) / (n_k[:, np.newaxis] + 1e-10)
        
        # 4. Compute adaptation coefficient (alpha_k)
        alpha_k = n_k / (n_k + relevance_factor)
        alpha_k = alpha_k[:, np.newaxis] # Reshape for broadcasting: (n_components, 1)
        
        # 5. MAP Mean Adaptation
        ubm_means = self.gmm.means_
        adapted_means = alpha_k * e_k + (1 - alpha_k) * ubm_means
        
        # 6. Flatten to create the supervector
        supervector = adapted_means.flatten()
        return supervector
