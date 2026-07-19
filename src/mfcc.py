"""
MFCC Feature Extraction Module

Pipeline:
Frames
    ↓
Hamming Window
    ↓
MFCC
    ↓
Delta
    ↓
Delta-Delta
"""

import librosa
import numpy as np


# ============================================================
# PARAMETERS
# ============================================================

N_MFCC = 13
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160          # 10 ms at 16 kHz
WIN_LENGTH = 400          # 25 ms at 16 kHz


# ============================================================
# MFCC
# ============================================================

def compute_mfcc(signal, sample_rate):
    """
    Compute MFCC features.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed audio signal

    sample_rate : int

    Returns
    -------
    np.ndarray
        Shape:
        (num_frames, 13)
    """

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS
    )

    return mfcc.T


# ============================================================
# DELTA
# ============================================================

def compute_delta(mfcc):
    """
    Compute first-order derivatives.
    """

    delta = librosa.feature.delta(mfcc.T)

    return delta.T


# ============================================================
# DELTA-DELTA
# ============================================================

def compute_delta_delta(mfcc):
    """
    Compute second-order derivatives.
    """

    delta2 = librosa.feature.delta(
        mfcc.T,
        order=2
    )

    return delta2.T


# ============================================================
# COMPLETE FEATURE EXTRACTION
# ============================================================

def extract_features(signal, sample_rate):
    """
    Extract complete MFCC feature vector.

    Output:

    13 MFCC
        +
    13 Delta
        +
    13 Delta-Delta

    = 39 Features
    """

    mfcc = compute_mfcc(
        signal,
        sample_rate
    )

    delta = compute_delta(mfcc)

    delta2 = compute_delta_delta(mfcc)

    features = np.concatenate(
        (
            mfcc,
            delta,
            delta2
        ),
        axis=1
    )

    # Apply CMVN
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)

    return features
if __name__ == "__main__":

    from preprocessing import preprocess_audio

    # CHANGE THIS PATH TO ANY ONE OF YOUR WAV FILES
    audio_path = r"D:\SLID_GMM_UBM\data\valid\audio\S4256073100322808_1.wav"

    signal, sr = preprocess_audio(audio_path)

    features = extract_features(signal, sr)

    print("=" * 50)
    print("MFCC FEATURE EXTRACTION SUCCESSFUL")
    print("=" * 50)

    print(f"Sampling Rate       : {sr}")
    print(f"Feature Shape       : {features.shape}")
    print(f"Number of Frames    : {features.shape[0]}")
    print(f"Features per Frame  : {features.shape[1]}")

    print("\nFirst Feature Vector:\n")
    print(features[0])