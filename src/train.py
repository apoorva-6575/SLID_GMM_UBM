"""
Master Orchestration Script.
Executes the entire SLID pipeline using the GMM-UBM Supervector architecture.
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
    If the .npy file already exists, it skips extraction to save time.
    
    Returns a list of feature matrices and a list of labels.
    """
    feature_dir = Path(data_dir) / "features" / split
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    audio_dir = Path(data_dir) / split / "audio"
    
    all_features = []
    all_labels = []
    
    print(f"\n--- Feature Extraction: {split.upper()} SET ---")
    
    for i, row in enumerate(metadata):
        wav_name = row['filepath']
        lang = row['language']
        
        wav_path = audio_dir / wav_name
        npy_path = feature_dir / wav_name.replace(".wav", ".npy")
        
        if npy_path.exists():
            # Load from cache
            features = np.load(npy_path)
        else:
            # Extract and save
            try:
                features = extract_features_from_file(wav_path)
                np.save(npy_path, features)
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")
                continue
                
        all_features.append(features)
        all_labels.append(lang)
        
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(metadata)} files.")
            
    print(f"Loaded {len(all_features)} feature matrices for {split}.")
    return all_features, all_labels


def main():
    print("=" * 60)
    print("SLID GMM-UBM SUPERVECTOR PIPELINE INITIALIZATION")
    print("=" * 60)
    
    config = load_config()
    name_to_label, label_to_name = get_language_mapping(config)
    
    data_dir = "d:\\SLID_GMM_UBM\\data"
    model_dir = Path("d:\\SLID_GMM_UBM\\models")
    model_dir.mkdir(exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. METADATA PARSING
    # ---------------------------------------------------------
    print("\n1. Parsing Metadata...")
    train_meta = parse_metadata('train', data_dir)
    test_meta = parse_metadata('test', data_dir)
    
    # Filter to only the languages defined in config
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
    # 3. PHASE 5: UBM TRAINING
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 5: UNIVERSAL BACKGROUND MODEL")
    print("=" * 60)
    
    ubm_dir = model_dir / "gmm_models"
    ubm_dir.mkdir(exist_ok=True)
    ubm_path = ubm_dir / "ubm.pkl"
    
    if ubm_path.exists():
        print("UBM already exists. Loading from disk...")
        ubm = UBM.load(ubm_path)
    else:
        print("Training UBM from scratch...")
        # Concatenate all training frames
        train_features_concat = np.vstack(X_train_list)
        
        MAX_UBM_FRAMES = 2_000_000
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
        
        # Free up memory
        del train_features_concat

    # ---------------------------------------------------------
    # 4. PHASE 6: SUPERVECTOR EXTRACTION
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 6: SUPERVECTOR EXTRACTION")
    print("=" * 60)
    
    def extract_supervectors_for_dataset(feature_list, ubm_model):
        """Extracts a MAP-adapted supervector for each utterance."""
        supervectors = []
        for i, features in enumerate(feature_list):
            sv = ubm_model.extract_supervector(features)
            supervectors.append(sv)
            if (i + 1) % 5000 == 0:
                print(f"Extracted {i + 1} supervectors...")
        return np.array(supervectors)

    print("Extracting Supervectors for Train set...")
    X_train_sv = extract_supervectors_for_dataset(X_train_list, ubm)
    
    # Normalize the supervectors
    scaler = StandardScaler()
    X_train_sv = scaler.fit_transform(X_train_sv)
    joblib.dump(scaler, model_dir / "ann_scaler.pkl")
    
    y_train_idx = np.array([name_to_label[l] for l in y_train_str])
    
    # ---------------------------------------------------------
    # 5. PHASE 7: ANN CLASSIFIER
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 7: ANN CLASSIFIER")
    print("=" * 60)
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train_sv)
    y_train_tensor = torch.LongTensor(y_train_idx)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['ann']['batch_size'], shuffle=True)
    
    ann_path = model_dir / "ann_model.pt"
    
    supervector_dim = X_train_sv.shape[1]
    print(f"Supervector Dimension: {supervector_dim}")
    
    ann_model = LanguageANN(
        input_dim=supervector_dim,
        num_classes=len(valid_langs),
        hidden_dims=config['ann']['hidden_dims'],
        dropout=config['ann']['dropout']
    )
    
    print("Training PyTorch ANN...")
    ann_model = train_ann(
        model=ann_model, 
        train_loader=train_loader, 
        epochs=config['ann']['epochs'], 
        learning_rate=config['ann']['learning_rate']
    )
    save_ann(ann_model, ann_path)

    # ---------------------------------------------------------
    # 6. PHASE 8: EVALUATION
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 8: EVALUATION")
    print("=" * 60)
    
    print("Extracting Supervectors for Test set...")
    X_test_sv = extract_supervectors_for_dataset(X_test_list, ubm)
    
    # Normalize using the saved scaler
    X_test_sv = scaler.transform(X_test_sv)
    
    y_test_idx = np.array([name_to_label[l] for l in y_test_str])
    
    X_test_tensor = torch.FloatTensor(X_test_sv)
    
    print("Running inference on Test set...")
    ann_model.eval() # Set to eval mode (disables dropout)
    with torch.no_grad():
        outputs = ann_model(X_test_tensor)
        _, predictions = torch.max(outputs.data, 1)
        y_pred = predictions.numpy()
        
    evaluate_model(y_test_idx, y_pred, label_to_name)


if __name__ == "__main__":
    main()
