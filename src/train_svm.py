import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from utils import load_config, get_language_mapping

def train_and_eval_svm():
    model_dir = Path("d:\\SLID_GMM_UBM\\models")
    sv_cache_dir = model_dir / "sv_cache_64_v2"
    
    print("Loading supervectors from cache...")
    X_train = np.load(sv_cache_dir / "X_train_sv.npy")
    y_train = np.load(sv_cache_dir / "y_train.npy")
    X_test = np.load(sv_cache_dir / "X_test_sv.npy")
    y_test = np.load(sv_cache_dir / "y_test.npy")
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Paper specifics: OvA (One-vs-Rest), Gaussian (RBF) kernel, regularization factor (C) 1.3
    print("Training SVM (RBF kernel, C=1.3, OvR)...")
    # In sklearn, SVC with decision_function_shape='ovr' handles One-vs-Rest natively
    svm_model = SVC(kernel='rbf', C=1.3, decision_function_shape='ovr', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    print("Evaluating SVM...")
    y_pred_train = svm_model.predict(X_train_scaled)
    y_pred_test = svm_model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\nTrain Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    
    config = load_config()
    _, label_to_name = get_language_mapping(config)
    target_names = [label_to_name[i] for i in range(len(label_to_name))]
    
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred_test, target_names=target_names))

if __name__ == "__main__":
    train_and_eval_svm()
