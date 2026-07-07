"""Audio loading utilities."""

from pathlib import Path

import librosa


def load_audio(path, sample_rate=16000, mono=True):
    """Load an audio file."""
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    signal, sr = librosa.load(audio_path, sr=sample_rate, mono=mono)
    return signal, sr
