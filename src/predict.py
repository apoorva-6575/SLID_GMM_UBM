"""
Inference Script.

This script takes a raw audio file as input, processes it through the entire
pipeline (MFCC -> UBM -> GMM -> StandardScaler -> ANN), and outputs the predicted language.

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
from gmm import LanguageGMM
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
    print("Loading models (UBM, GMMs, Scaler, ANN)...")
    
    # Load UBM
    ubm_path = model_dir / "gmm_models" / "ubm.pkl"
    ubm = UBM.load(ubm_path)
    
    # Load GMMs
    language_gmms = {}
    for lang in valid_langs:
        gmm_path = model_dir / "gmm_models" / f"{lang}_gmm.pkl"
        language_gmms[lang] = LanguageGMM.load(gmm_path, lang)
        
    # Load Scaler
    scaler_path = model_dir / "ann_scaler.pkl"
    scaler = joblib.load(scaler_path)
    
    # Load PyTorch ANN
    ann_path = model_dir / "ann_model.pt"
    ann_model = LanguageANN(
        input_dim=len(valid_langs),
        num_classes=len(valid_langs),
        hidden_dims=config['ann']['hidden_dims'],
        dropout=config['ann']['dropout']
    )
    ann_model = load_ann(ann_model, ann_path)
    
    # 4. Generate GMM Scores
    print("Scoring against language GMMs...")
    scores = []
    ordered_langs = [label_to_name[i] for i in range(len(label_to_name))]
    
    for lang in ordered_langs:
        if lang in language_gmms:
            scores.append(language_gmms[lang].score(features))
        else:
            scores.append(-9999.0)
            
    # Reshape for scalar/model (1 sample, N features)
    score_vector = np.array([scores])
    
    # 5. Normalize Scores
    normalized_scores = scaler.transform(score_vector)
    
    # 6. ANN Prediction
    score_tensor = torch.FloatTensor(normalized_scores)
    
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
    for i, lang in enumerate(ordered_langs):
        print(f"  - {lang.capitalize():<10} : {probabilities[i]*100:>6.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_wav_file>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    predict_language(audio_file)
