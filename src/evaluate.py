"""Evaluation utilities."""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_predictions(y_true, y_pred):
    """Compute standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
