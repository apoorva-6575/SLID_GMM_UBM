"""Audio preprocessing helpers."""

from pathlib import Path

import librosa
import numpy as np


def normalize_audio(signal):
    """Peak-normalize an audio signal."""
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal
    return signal / peak


def resample_audio(signal, original_sr, target_sr=16000):
    """Resample audio to a target sampling rate."""
    if original_sr == target_sr:
        return signal
    return librosa.resample(signal, orig_sr=original_sr, target_sr=target_sr)


def load_audio(path, sample_rate=16000, mono=True):
    """Load an audio file.

    Parameters
    ----------
    path : str or Path
        Path to the audio file.
    sample_rate : int
        Target sampling rate.
    mono : bool
        Whether to mix down to mono.

    Returns
    -------
    tuple of (np.ndarray, int)
        Audio signal and sampling rate.
    """
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    signal, sr = librosa.load(audio_path, sr=sample_rate, mono=mono)
    return signal, sr
