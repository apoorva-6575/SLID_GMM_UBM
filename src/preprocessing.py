"""
Audio preprocessing helpers for Spoken Language Identification (SLID).
"""

from pathlib import Path

import librosa
import numpy as np

TARGET_SR = 16000


def load_audio(path):
    """Load audio without modifying it."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    signal, sr = librosa.load(
        path,
        sr=None,
        mono=False
    )

    return signal, sr


def to_mono(signal):
    """Convert stereo audio to mono."""

    if signal.ndim == 1:
        return signal

    return librosa.to_mono(signal)


def resample_audio(signal, original_sr, target_sr=TARGET_SR):
    """Resample audio."""

    if original_sr == target_sr:
        return signal, original_sr

    signal = librosa.resample(
        signal,
        orig_sr=original_sr,
        target_sr=target_sr
    )

    return signal, target_sr


def trim_silence(signal):
    """Trim leading and trailing silence."""

    signal, _ = librosa.effects.trim(
        signal,
        top_db=20
    )

    return signal


def normalize_audio(signal):
    """Peak normalize."""

    peak = np.max(np.abs(signal))

    if peak == 0:
        return signal

    return signal / peak


def preprocess_audio(path):
    """
    Complete preprocessing pipeline.

    Returns
    -------
    signal : np.ndarray
    sr : int
    """

    signal, sr = load_audio(path)

    signal = to_mono(signal)

    signal, sr = resample_audio(signal, sr)

    signal = trim_silence(signal)

    signal = normalize_audio(signal)

    return signal, sr