"""
Evaluation module for IoMT intrusion detection.
Owner: Karen Nyadu
Receives y_test + CNN predictions + Isolation Forest predictions.
Produces metrics, confusion matrices, and results/metrics.csv.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred, model_name):
    """Compute the full metric set for one model."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # False negative rate: missed attacks / all real attacks
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "model": model_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_negative_rate": fnr,
        "accuracy": accuracy_score(y_true, y_pred),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def plot_confusion_matrix(y_true, y_pred, model_name, out_path):
    """Save a labeled confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Malicious/Suspicious"])
    ax.set_yticklabels(["Normal", "Malicious/Suspicious"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_all(y_test, cnn_pred, if_pred,
                 out_csv="results/metrics.csv",
                 out_dir="results"):
    """
    Evaluate both models on the same held-out rows.

    Args:
        y_test:    true labels (0=benign, 1=malicious)
        cnn_pred:  CNN binary predictions (0/1), same row order as y_test
        if_pred:   Isolation Forest predictions (0=normal, 1=suspicious),
                   same row order as y_test
        out_csv:   where to write the metrics table
        out_dir:   folder for confusion matrix figures
    """
    # 1) Verify alignment — the handbook's first step
    assert len(y_test) == len(cnn_pred) == len(if_pred), \
        "Prediction row counts do not match y_test"

    # 2) Evaluate separately
    rows = [
        compute_metrics(y_test, cnn_pred, "CNN"),
        compute_metrics(y_test, if_pred, "IsolationForest"),
    ]

    # 3) Save metrics CSV
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # 4) Confusion matrix figures
    plot_confusion_matrix(y_test, cnn_pred, "CNN",
                          os.path.join(out_dir, "cm_cnn.png"))
    plot_confusion_matrix(y_test, if_pred, "IsolationForest",
                          os.path.join(out_dir, "cm_isolation_forest.png"))

    return pd.DataFrame(rows)