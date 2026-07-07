"""Audio preprocessing helpers."""

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
