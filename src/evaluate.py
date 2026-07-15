"""
Evaluation Module.
Computes and prints standard classification metrics (Accuracy,
Confusion Matrix, Classification Report).
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np


def evaluate_model(y_true, y_pred, label_to_name):
    """
    Calculate and print evaluation metrics.
    
    Parameters
    ----------
    y_true : list or np.ndarray
        True integer labels.
    y_pred : list or np.ndarray
        Predicted integer labels.
    label_to_name : dict
        Mapping from integer label to language name.
    """
    
    print("\n" + "=" * 50)
    print("PHASE 8: EVALUATION RESULTS")
    print("=" * 50)
    
    # 1. Overall Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc * 100:.2f}%\n")
    
    # Target names in correct integer order
    target_names = [label_to_name[i] for i in range(len(label_to_name))]
    
    # 2. Classification Report (Precision, Recall, F1)
    # We use zero_division=0 to prevent warnings if a class is entirely missing in predictions
    print("Classification Report:")
    labels = list(range(len(target_names)))
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0))
    
    # 3. Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    
    # Print formatted confusion matrix
    header = f"{'True \\ Pred':<15}" + "".join([f"{name[:3]:>8}" for name in target_names])
    print(header)
    print("-" * len(header))
    
    for i, row_label in enumerate(target_names):
        row_str = f"{row_label:<15}"
        for val in cm[i]:
            row_str += f"{val:>8}"
        print(row_str)
        
    print("\nNote: Rows represent the TRUE language, Columns represent the PREDICTED language.")
    print("=" * 50)
