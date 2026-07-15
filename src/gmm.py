"""
Language-specific Gaussian Mixture Model (GMM) Module.

This module adapts a pre-trained Universal Background Model (UBM)
to specific language data using EM initialization (a practical alternative
to strict MAP adaptation when using scikit-learn).
"""

from pathlib import Path

import joblib
from sklearn.mixture import GaussianMixture


class LanguageGMM:
    """
    Language-specific GMM adapted from a UBM.
    """

    def __init__(self, language_name, ubm=None, max_iter=20):
        """
        Initialize the Language GMM.

        Parameters
        ----------
        language_name : str
            Name of the language (e.g., 'hindi').
        ubm : UBM object
            The pre-trained Universal Background Model to adapt from.
            If None, it must be loaded from disk later.
        max_iter : int
            Number of EM iterations for adaptation. Kept small (e.g., 20)
            so the model adapts but doesn't completely overwrite the UBM structure.
        """
        self.language_name = language_name
        self.is_trained = False
        
        if ubm is not None:
            if not ubm.is_trained:
                raise ValueError("The provided UBM is not trained.")
                
            # Initialize a new GMM with the exact parameters of the UBM
            self.gmm = GaussianMixture(
                n_components=ubm.gmm.n_components,
                covariance_type=ubm.gmm.covariance_type,
                max_iter=max_iter,
                random_state=ubm.gmm.random_state,
                weights_init=ubm.gmm.weights_,
                means_init=ubm.gmm.means_,
                precisions_init=ubm.gmm.precisions_cholesky_, # sklearn uses cholesky precision internally
                verbose=2,
                verbose_interval=5
            )
        else:
            self.gmm = None

    def train(self, features):
        """
        Adapt the GMM to the language-specific features.

        Parameters
        ----------
        features : np.ndarray
            Matrix of concatenated MFCC features for this specific language.
            Shape: (total_num_frames, num_features)
        """
        if self.gmm is None:
            raise RuntimeError("GMM was not initialized with a UBM.")
            
        if len(features.shape) != 2:
            raise ValueError(f"Features must be a 2D array. Got shape {features.shape}")

        print(f"Adapting {self.language_name.capitalize()} GMM on {features.shape[0]} frames...")
        
        # The fit() method will start from the UBM parameters (due to the *_init arguments)
        # and run EM on the language features.
        self.gmm.fit(features)
        
        self.is_trained = True
        print(f"{self.language_name.capitalize()} GMM adaptation complete.")

    def score(self, features):
        """
        Calculate the log-likelihood of a feature matrix under this GMM.
        Used during evaluation to find the most likely language.
        
        Returns the average log-likelihood per frame.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot score with an untrained GMM.")
            
        return self.gmm.score(features)

    def save(self, output_dir):
        """Save the adapted GMM to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained GMM.")
        
        path = Path(output_dir) / f"{self.language_name}_gmm.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.gmm, path)
        print(f"GMM saved to {path}")

    @classmethod
    def load(cls, filepath, language_name):
        """Load a trained language GMM from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        instance = cls(language_name=language_name)
        instance.gmm = joblib.load(path)
        instance.is_trained = True
        return instance
