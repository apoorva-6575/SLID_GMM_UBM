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
    
    # Save the results to file system
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    metrics_dir = "d:\\SLID_GMM_UBM\\results\\metrics"
    cm_dir = "d:\\SLID_GMM_UBM\\results\\confusion_matrix"
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    
    # Save classification report and accuracy
    metrics_path = os.path.join(metrics_dir, "classification_report.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0))
    print(f"Saved classification metrics to {metrics_path}")
        
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('True Language')
    plt.xlabel('Predicted Language')
    plt.title('Language Identification Confusion Matrix')
    cm_path = os.path.join(cm_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

if __name__ == "__main__":
    # When run directly, just print the most recently saved metrics
    import os
    metrics_path = "d:\\SLID_GMM_UBM\\results\\metrics\\classification_report.txt"
    if os.path.exists(metrics_path):
        print("\n" + "=" * 50)
        print("LATEST EVALUATION RESULTS")
        print("=" * 50 + "\n")
        with open(metrics_path, "r") as f:
            print(f.read())
        print("=" * 50)
        print("To view the confusion matrix, open: d:\\SLID_GMM_UBM\\results\\confusion_matrix\\confusion_matrix.png")
    else:
        print("No evaluation results found. Please run src/train.py first.")

