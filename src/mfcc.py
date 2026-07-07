"""MFCC extraction utilities."""

import librosa
import numpy as np


def extract_mfcc(signal, sample_rate=16000, n_mfcc=13, n_fft=400, hop_length=160):
    """Extract MFCC features as rows of frames."""
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return mfcc.T


def append_delta_features(mfcc_features):
    """Append delta and delta-delta coefficients."""
    delta = librosa.feature.delta(mfcc_features.T).T
    delta_delta = librosa.feature.delta(mfcc_features.T, order=2).T
    return np.concatenate([mfcc_features, delta, delta_delta], axis=1)
