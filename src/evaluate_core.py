from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def safe_proba(model, X):
    # Some models might not have predict_proba
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
    # fallback: decision function -> sigmoid-ish scaling not guaranteed
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        # min-max to [0,1] for ranking-based metrics (not calibration)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return s
    # last resort: predicted labels
    return model.predict(X)

def compute_metrics(y_true, y_pred, y_score) -> dict:
    out = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    # AUC only valid when both classes present
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["avg_precision"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
        out["avg_precision"] = float("nan")
    return out

def confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm
