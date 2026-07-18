"""
Inference Script.

This script takes a raw audio file as input, processes it through the entire
pipeline (MFCC -> UBM Supervector -> StandardScaler -> ANN), and outputs the predicted language.

Usage:
    python src/predict.py path/to/audio.wav
"""

import sys
import os
from pathlib import Path

import numpy as np
import torch
import joblib

# Import pipeline modules
from utils import load_config, get_language_mapping
from feature_extraction import extract_features_from_file
from ubm import UBM
from ann import LanguageANN, load_ann

# Ensure PyTorch doesn't crash on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def predict_language(audio_path):
    """Run inference on a single audio file."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"Error: File not found at {audio_path}")
        sys.exit(1)
        
    print(f"Processing audio: {audio_path.name}")
    
    # 1. Load config and mappings
    config = load_config()
    name_to_label, label_to_name = get_language_mapping(config)
    valid_langs = list(name_to_label.keys())
    
    model_dir = Path("d:\\SLID_GMM_UBM\\models")
    
    # 2. Extract Features
    print("Extracting MFCC features...")
    try:
        features = extract_features_from_file(audio_path)
    except Exception as e:
        print(f"Failed to extract features: {e}")
        sys.exit(1)
        
    # 3. Load Models
    print("Loading models (UBM, Scaler, ANN)...")
    
    # Load UBM
    ubm_path = model_dir / "gmm_models" / "ubm.pkl"
    ubm = UBM.load(ubm_path)
    
    # Load Scaler
    scaler_path = model_dir / "ann_scaler.pkl"
    scaler = joblib.load(scaler_path)
    
    # 4. Generate Supervector
    print("Extracting Supervector...")
    supervector = ubm.extract_supervector(features)
    
    # Load PyTorch ANN dynamically based on supervector shape
    ann_path = model_dir / "ann_model.pt"
    ann_model = LanguageANN(
        input_dim=supervector.shape[0],
        num_classes=len(valid_langs),
        hidden_dims=config['ann']['hidden_dims'],
        dropout=config['ann']['dropout']
    )
    ann_model = load_ann(ann_model, ann_path)
    
    # Reshape for scalar/model (1 sample, N features)
    sv_matrix = np.array([supervector])
    
    # 5. Normalize Scores
    normalized_sv = scaler.transform(sv_matrix)
    
    # 6. ANN Prediction
    score_tensor = torch.FloatTensor(normalized_sv)
    
    with torch.no_grad():
        logits = ann_model(score_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
        
    # 7. Print Results
    predicted_idx = np.argmax(probabilities)
    predicted_lang = label_to_name[predicted_idx]
    
    print("\n" + "=" * 40)
    print(f"PREDICTED LANGUAGE: {predicted_lang.upper()}")
    print("=" * 40)
    print("Confidence Scores:")
    ordered_langs = [label_to_name[i] for i in range(len(label_to_name))]
    for i, lang in enumerate(ordered_langs):
        print(f"  - {lang.capitalize():<10} : {probabilities[i]*100:>6.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_wav_file>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    predict_language(audio_file)
