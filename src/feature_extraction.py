"""
Complete Feature Extraction Pipeline

Audio file → Preprocessing → MFCC + Delta + Delta-Delta → Feature matrix

Note:
    framing.py contains a manual framing implementation for educational
    purposes. However, librosa.feature.mfcc() performs its own internal
    framing and windowing, so we call extract_features() directly on the
    preprocessed signal.
"""

from pathlib import Path

from preprocessing import preprocess_audio
from mfcc import extract_features


def extract_features_from_file(audio_path):
    """
    Complete pipeline for one audio file.

    Parameters
    ----------
    audio_path : str or Path

    Returns
    -------
    np.ndarray
        Shape:
        (num_frames, 39)
    """

    audio_path = Path(audio_path)

    signal, sample_rate = preprocess_audio(audio_path)

    features = extract_features(
        signal,
        sample_rate
    )

    return features