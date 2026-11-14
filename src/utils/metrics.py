import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


def calculate_classification_metrics(y_true, y_pred, target_names=None):
    """Tinh toan cac chi so danh gia da lop."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
        "report_text": report
    }

