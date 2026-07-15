"""
Framing utilities for Spoken Language Identification.
"""

import numpy as np


FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10


def frame_signal(signal, sample_rate):
    """
    Split signal into overlapping frames.

    Parameters
    ----------
    signal : np.ndarray
        Audio signal.
    sample_rate : int
        Sampling rate.

    Returns
    -------
    np.ndarray
        Shape = (num_frames, frame_length)
    """

    frame_length = int(sample_rate * FRAME_LENGTH_MS / 1000)
    frame_step = int(sample_rate * FRAME_SHIFT_MS / 1000)

    signal_length = len(signal)

    num_frames = int(
        np.ceil(
            (signal_length - frame_length) / frame_step
        )
    ) + 1

    pad_length = (num_frames - 1) * frame_step + frame_length

    zeros = np.zeros(pad_length - signal_length)

    padded_signal = np.concatenate((signal, zeros))

    indices = (
        np.tile(np.arange(frame_length), (num_frames, 1))
        +
        np.tile(
            np.arange(0, num_frames * frame_step, frame_step),
            (frame_length, 1)
        ).T
    )

    frames = padded_signal[indices.astype(np.int32)]

    return frames
def apply_hamming_window(frames):
    """
    Apply a Hamming window to every frame.
    """

    window = np.hamming(frames.shape[1])

    return frames * window