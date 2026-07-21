import os
import sys
sys.path.append("src")

import pickle
import numpy as np
import torch
import shutil
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import load_config, get_language_mapping
from src.feature_extraction import extract_features_from_file
from src.ubm import UBM
from src.ann import LanguageANN, load_ann

def evaluate_and_save(y_true, y_pred, label_to_name, split_name, out_dir):
    acc = accuracy_score(y_true, y_pred)
    target_names = [label_to_name[i] for i in range(len(label_to_name))]
    labels = list(range(len(target_names)))
    
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    
    # Save Report
    with open(os.path.join(out_dir, f"{split_name}_classification_report.txt"), "w") as f:
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    # Save CM
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('True Language')
    plt.xlabel('Predicted Language')
    plt.title(f'{split_name.capitalize()} Set - Confusion Matrix')
    plt.savefig(os.path.join(out_dir, f"{split_name}_confusion_matrix.png"))
    plt.close()
    
    print(f"[{split_name.upper()}] Accuracy: {acc * 100:.2f}%")

def main():
    print("Generating Mentor Artifacts...")
    config = load_config("configs/config.yaml")
    name_to_label, label_to_name = get_language_mapping(config)
    
    model_dir = Path("models")
    sv_dir = model_dir / "sv_cache_64_v2"
    sv_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path("data")
    
    # 1. Ensure validation features are extracted and supervectors are cached
    valid_sv_path = sv_dir / "X_valid_sv.npy"
    valid_y_path = sv_dir / "y_valid.npy"
    
    if not valid_sv_path.exists():
        print("Extracting Supervectors for Validation Set (this may take a minute)...")
        ubm_path = model_dir / "gmm_models" / "ubm_64_v2.pkl"
        ubm = UBM.load(ubm_path)
        
        valid_audio_dir = data_dir / "valid" / "audio"
        X_valid_sv = []
        y_valid = []
        
        audio_files = list(valid_audio_dir.glob("*.wav"))
        for i, path in enumerate(audio_files):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(audio_files)}")
            lang = path.name.split('_')[0]
            if lang in name_to_label:
                try:
                    features = extract_features_from_file(path)
                    sv = ubm.extract_supervector(features)
                    X_valid_sv.append(sv)
                    y_valid.append(name_to_label[lang])
                except Exception as e:
                    pass
        
        X_valid_sv = np.array(X_valid_sv)
        y_valid = np.array(y_valid)
        np.save(valid_sv_path, X_valid_sv)
        np.save(valid_y_path, y_valid)
    else:
        print("Validation supervectors already cached.")
        X_valid_sv = np.load(valid_sv_path)
        y_valid = np.load(valid_y_path)
        
    # 2. Load Train and Test
    X_train_sv = np.load(sv_dir / "X_train_sv.npy")
    y_train = np.load(sv_dir / "y_train.npy")
    X_test_sv = np.load(sv_dir / "X_test_sv.npy")
    y_test = np.load(sv_dir / "y_test.npy")
    
    # 3. Load ANN and Scaler
    import joblib
    scaler = joblib.load(model_dir / "ann_scaler.pkl")
    
    ann_model = LanguageANN(
        input_dim=X_train_sv.shape[1],
        num_classes=len(name_to_label),
        hidden_dims=config['ann']['hidden_dims']
    )
    ann_model = load_ann(ann_model, model_dir / "ann_model.pt")
    
    # 4. Predict
    def get_predictions(X_already_scaled):
        with torch.no_grad():
            logits = ann_model(torch.FloatTensor(X_already_scaled))
            _, preds = torch.max(logits, 1)
        return preds.numpy()
        
    print("\nRunning Inference...")
    y_train_pred = get_predictions(X_train_sv)
    y_valid_pred = get_predictions(scaler.transform(X_valid_sv))
    y_test_pred = get_predictions(X_test_sv)
    
    # 5. Save Results
    out_dir = Path("mentor_deliverables")
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    evaluate_and_save(y_train, y_train_pred, label_to_name, "train", metrics_dir)
    evaluate_and_save(y_valid, y_valid_pred, label_to_name, "valid", metrics_dir)
    evaluate_and_save(y_test, y_test_pred, label_to_name, "test", metrics_dir)
    
    # 6. Prepare Sample Audio
    print("Preparing Sample Audio...")
    sample_audio_dir = out_dir / "sample_audio"
    sample_audio_dir.mkdir(parents=True, exist_ok=True)
    
    for lang in name_to_label.keys():
        # Get one test file
        test_files = list((data_dir / "test" / "audio").glob(f"{lang}_*.wav"))
        if test_files:
            shutil.copy(test_files[0], sample_audio_dir / test_files[0].name)
            
    # Copy scripts
    shutil.copy("src/predict.py", out_dir / "predict.py")
    
    # Create zip
    print("Creating ZIP file...")
    shutil.make_archive("mentor_deliverables", 'zip', out_dir)
    
    print("\nSUCCESS! mentor_deliverables.zip has been created.")
    
if __name__ == "__main__":
    main()
