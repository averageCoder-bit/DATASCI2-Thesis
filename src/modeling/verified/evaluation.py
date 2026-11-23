"""Read-only label alignment and test evaluation for the verified experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def align_proxy_labels(labels: pd.DataFrame, sequence_ids) -> np.ndarray:
    """Require a one-to-one relationship to the recorded sequence ordering.

    Prevents threshold drift caused by pairing correct arrays in the wrong order.
    Returns aligned 1D int8 binary proxy labels; does not fit the model.
    """
    ids = np.asarray(sequence_ids)
    if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("Sequence IDs must be a nonempty unique 1D array")
    needed = {"sequence_id", "proxy_anomaly"}
    if not needed.issubset(labels.columns):
        raise ValueError(f"Missing proxy columns: {sorted(needed-set(labels.columns))}")
    if labels["sequence_id"].isna().any() or labels["sequence_id"].duplicated().any():
        raise ValueError("Proxy-label sequence IDs are missing or duplicated")
    aligned = pd.DataFrame({"sequence_id": ids}).merge(
        labels[["sequence_id", "proxy_anomaly"]], on="sequence_id",
        how="left", sort=False, validate="one_to_one",
    )
    if len(aligned) != len(ids) or aligned.proxy_anomaly.isna().any():
        raise ValueError("One or more sequence IDs are not covered by proxy labels")
    y = pd.to_numeric(aligned.proxy_anomaly, errors="raise").to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Proxy labels must be binary 0/1")
    return y.astype(np.int8)


def evaluate_heldout(scores, labels, frozen_threshold: float) -> dict:
    """Evaluate held-out scores against proxy labels; never fit on held-out data."""
    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Invalid test scores")
    if y.shape != values.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be aligned binary 0/1 values")
    y = y.astype(np.int8)
    if len(np.unique(y)) != 2:
        raise ValueError("Metrics require both proxy classes")
    if not np.isfinite(frozen_threshold):
        raise ValueError("Threshold must be finite")
    predictions = (values >= frozen_threshold).astype(np.int8)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, predictions, labels=[0,1]).ravel())
    return {
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "roc_auc": float(roc_auc_score(y, values)),
        "average_precision": float(average_precision_score(y, values)),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "predicted_positive": int(predictions.sum()),
        "proxy_positive": int(y.sum()),
    }
