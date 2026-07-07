"""End-to-end feature extraction."""

from audio_loader import load_audio
from mfcc import append_delta_features, extract_mfcc
from preprocessing import normalize_audio


def extract_features_from_file(path, sample_rate=16000, n_mfcc=13):
    """Load audio and extract MFCC plus delta features."""
    signal, sr = load_audio(path, sample_rate=sample_rate)
    signal = normalize_audio(signal)
    mfcc_features = extract_mfcc(signal, sample_rate=sr, n_mfcc=n_mfcc)
    return append_delta_features(mfcc_features)
