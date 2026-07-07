"""Framing and windowing utilities."""

import librosa
import numpy as np


def frame_audio(signal, frame_length=400, hop_length=160):
    """Split audio into overlapping frames."""
    return librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length).T


def apply_hamming_window(frames):
    """Apply a Hamming window to framed audio."""
    window = np.hamming(frames.shape[1])
    return frames * window
