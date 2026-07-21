"""
Master Orchestration Script for 5-second audio pipeline.
Executes the SLID pipeline using the 5-second datasets.
"""

import os
import sys
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

# Import pipeline modules
from utils import load_config, get_language_mapping, parse_metadata
from feature_extraction import extract_features_from_file
from ubm import UBM
from ann import LanguageANN, train_ann, save_ann
from evaluate import evaluate_model

# Ignore scikit-learn warnings about early convergence in small datasets
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure PyTorch doesn't crash due to OpenMP duplicate libs on Conda/Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def extract_and_cache_features(split, metadata, data_dir):
    """
    Extracts features for all files in the metadata and saves them to disk as .npy.
    """
    feature_dir = Path(data_dir) / "features_5s" / split
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    audio_dir = Path(data_dir) / split / "audio_5s"
    
    all_features = []
    all_labels = []
    
    print(f"\n--- Feature Extraction (5s): {split.upper()} SET ---")
    
    for i, row in enumerate(metadata):
        wav_name = row['filepath']
        lang = row['language']
        
        wav_path = audio_dir / wav_name
        npy_path = feature_dir / wav_name.replace(".wav", ".npy")
        
        if npy_path.exists():
            features = np.load(npy_path)
        else:
            try:
                features = extract_features_from_file(wav_path)
                np.save(npy_path, features)
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")
                continue
                
        all_features.append(features)
        all_labels.append(lang)
        
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(metadata)} 5s files.")
            
    print(f"Loaded {len(all_features)} feature matrices for {split}.")
    return all_features, all_labels

def main():
    print("=" * 60)
    print("SLID 5-SECOND PIPELINE INITIALIZATION")
    print("=" * 60)
    
    config = load_config()
    name_to_label, label_to_name = get_language_mapping(config)
    
    data_dir = "d:\\SLID_GMM_UBM\\data"
    model_dir = Path("d:\\SLID_GMM_UBM\\models")
    results_dir = "d:\\SLID_GMM_UBM\\results_5s"
    
    model_dir.mkdir(exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. METADATA PARSING (5s)
    # ---------------------------------------------------------
    print("\n1. Parsing 5s Metadata...")
    # parse_metadata reads metadata.csv, so we will manually read metadata_5s.csv
    import csv
    def read_5s_meta(split):
        meta_path = Path(data_dir) / split / "metadata_5s.csv"
        rows = []
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        return rows
        
    train_meta = read_5s_meta('train')
    test_meta = read_5s_meta('test')
    
    valid_langs = set(name_to_label.keys())
    train_meta = [row for row in train_meta if row['language'] in valid_langs]
    test_meta = [row for row in test_meta if row['language'] in valid_langs]
    
    # ---------------------------------------------------------
    # 2. FEATURE EXTRACTION & CACHING
    # ---------------------------------------------------------
    print("\n2. Feature Extraction...")
    X_train_list, y_train_str = extract_and_cache_features('train', train_meta, data_dir)
    X_test_list, y_test_str = extract_and_cache_features('test', test_meta, data_dir)
    
    # ---------------------------------------------------------
    # 3. PHASE 5: UBM TRAINING (5s model)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 5: UNIVERSAL BACKGROUND MODEL (5s)")
    print("=" * 60)
    
    ubm_dir = model_dir / "gmm_models"
    ubm_dir.mkdir(exist_ok=True)
    ubm_path = ubm_dir / "ubm_64_5s.pkl"
    
    if ubm_path.exists():
        print("UBM 5s already exists. Loading from disk...")
        ubm = UBM.load(ubm_path)
    else:
        print("Training UBM 5s from scratch...")
        train_features_concat = np.vstack(X_train_list)
        
        MAX_UBM_FRAMES = 1_000_000
        if train_features_concat.shape[0] > MAX_UBM_FRAMES:
            print(f"Subsampling from {train_features_concat.shape[0]:,} to {MAX_UBM_FRAMES:,} frames...")
            rng = np.random.RandomState(42)
            indices = rng.choice(train_features_concat.shape[0], MAX_UBM_FRAMES, replace=False)
            train_features_concat = train_features_concat[indices]
        
        ubm = UBM(
            n_components=config['ubm']['n_components'],
            max_iter=config['ubm']['max_iter']
        )
        ubm.train(train_features_concat)
        ubm.save(ubm_path)
        del train_features_concat

    # ---------------------------------------------------------
    # 4. PHASE 6: SUPERVECTOR EXTRACTION (5s)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 6: SUPERVECTOR EXTRACTION (5s)")
    print("=" * 60)
    
    sv_cache_dir = model_dir / "sv_cache_64_5s"
    sv_cache_dir.mkdir(exist_ok=True)
    
    train_sv_path = sv_cache_dir / "X_train_sv_5s.npy"
    train_labels_path = sv_cache_dir / "y_train_5s.npy"
    test_sv_path = sv_cache_dir / "X_test_sv_5s.npy"
    test_labels_path = sv_cache_dir / "y_test_5s.npy"
    
    def extract_supervectors_for_dataset(feature_list, ubm_model):
        supervectors = []
        for i, features in enumerate(feature_list):
            sv = ubm_model.extract_supervector(features)
            supervectors.append(sv)
            if (i + 1) % 5000 == 0:
                print(f"Extracted {i + 1} supervectors...")
        return np.array(supervectors)

    y_train_idx = np.array([name_to_label[l] for l in y_train_str])
    y_test_idx = np.array([name_to_label[l] for l in y_test_str])

    if train_sv_path.exists() and test_sv_path.exists():
        print("Loading cached 5s supervectors from disk...")
        X_train_sv = np.load(train_sv_path)
        X_test_sv = np.load(test_sv_path)
        scaler = joblib.load(model_dir / "ann_scaler_5s.pkl")
        print(f"Loaded {X_train_sv.shape[0]} train and {X_test_sv.shape[0]} test supervectors.")
    else:
        print("Extracting Supervectors for Train set...")
        X_train_sv = extract_supervectors_for_dataset(X_train_list, ubm)
        
        scaler = StandardScaler()
        X_train_sv = scaler.fit_transform(X_train_sv)
        joblib.dump(scaler, model_dir / "ann_scaler_5s.pkl")
        
        print("Extracting Supervectors for Test set...")
        X_test_sv = extract_supervectors_for_dataset(X_test_list, ubm)
        X_test_sv = scaler.transform(X_test_sv)
        
        np.save(train_sv_path, X_train_sv)
        np.save(train_labels_path, y_train_idx)
        np.save(test_sv_path, X_test_sv)
        np.save(test_labels_path, y_test_idx)
        print("Supervectors cached to disk.")
    
    # ---------------------------------------------------------
    # 5. PHASE 7: ANN CLASSIFIER (5s)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 7: ANN CLASSIFIER (5s)")
    print("=" * 60)
    
    X_train_tensor = torch.FloatTensor(X_train_sv)
    y_train_tensor = torch.LongTensor(y_train_idx)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['ann']['batch_size'], shuffle=True)
    
    ann_path = model_dir / "ann_model_5s.pt"
    
    supervector_dim = X_train_sv.shape[1]
    print(f"Supervector Dimension: {supervector_dim}")
    
    ann_model = LanguageANN(
        input_dim=supervector_dim,
        num_classes=len(valid_langs),
        hidden_dims=config['ann']['hidden_dims'],
        dropout=config['ann']['dropout']
    )
    
    print("Training PyTorch ANN on 5s chunks...")
    ann_model = train_ann(
        model=ann_model, 
        train_loader=train_loader, 
        epochs=config['ann']['epochs'], 
        learning_rate=config['ann']['learning_rate']
    )
    save_ann(ann_model, ann_path)

    # ---------------------------------------------------------
    # 6. PHASE 8: EVALUATION (5s)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 8: EVALUATION (5s)")
    print("=" * 60)
    
    X_test_tensor = torch.FloatTensor(X_test_sv)
    
    print("Running inference on Test set...")
    ann_model.eval()
    with torch.no_grad():
        outputs = ann_model(X_test_tensor)
        _, predictions = torch.max(outputs.data, 1)
        y_pred = predictions.numpy()
        
    evaluate_model(y_test_idx, y_pred, label_to_name, results_dir=results_dir)

if __name__ == "__main__":
    main()
