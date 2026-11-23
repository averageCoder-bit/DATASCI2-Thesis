"""Pure GMM/BIC selection and validation-only threshold calibration utilities.

Extracted from the reviewed Phase 2 reproduction methodology. No file writes,
no test data used in model fitting or threshold selection.
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from .config import GMM_COMPONENTS, GMM_KWARGS


def _as_mae(errors, name: str = "MAE") -> np.ndarray:
    a = np.asarray(errors)
    if a.ndim != 1 or a.size == 0 or a.dtype.kind not in "fi":
        raise ValueError(f"{name} must be a nonempty 1D numeric array")
    a = a.astype(np.float64)
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError(f"{name} contains invalid reconstruction errors")
    return a


def fit_gmm_candidates(training_mae):
    """Fit recorded K=2..5 full-covariance GMM candidates; minimum BIC wins.

    Returns (selected_fitted_model, ordered_model_selection_rows).
    Runs only when explicitly invoked; never saves over frozen artifacts.
    """
    train = _as_mae(training_mae, "training MAE").reshape(-1, 1)
    fitted = {}
    rows = []
    for k in GMM_COMPONENTS:
        model = GaussianMixture(n_components=k, **GMM_KWARGS)
        model.fit(train)
        if not model.converged_:
            raise RuntimeError(f"GMM K={k} did not converge")
        rows.append({
            "n_components": int(k),
            "bic": float(model.bic(train)),
            "aic": float(model.aic(train)),
            "average_log_likelihood": float(model.score(train)),
            "converged": bool(model.converged_),
            "iterations": int(model.n_iter_),
        })
        fitted[k] = model
    # Python min preserves order in a tie (lowest K tested first).
    best_k = min(rows, key=lambda row: row["bic"])["n_components"]
    return fitted[best_k], rows


def negative_log_density(gmm, reconstruction_mae):
    """Return -ln p(MAE | fitted GMM); density may exceed 1, giving negative scores."""
    errors = _as_mae(reconstruction_mae)
    scores = -gmm.score_samples(errors.reshape(-1, 1))
    if scores.shape != errors.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid GMM anomaly scores")
    return scores


def select_validation_threshold(scores, labels):
    """Maximum validation balanced accuracy with recorded >= and first-tie rule.

    O(n log n) equivalent to ascending unique-score exhaustive search.
    An equal-length check is not sufficient to verify label/score row alignment;
    use `align_proxy_labels` before calling this function.
    """
    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite, nonempty 1D array")
    if y.shape != values.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be aligned binary 0/1 values")
    y = y.astype(np.int64)
    order = np.argsort(values, kind="stable")
    sorted_scores = values[order]
    sorted_y = y[order]
    positives = int(sorted_y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Validation labels must contain both classes")
    unique_scores, starts = np.unique(sorted_scores, return_index=True)
    pos_below = np.r_[0, np.cumsum(sorted_y, dtype=np.int64)][starts]
    neg_below = starts - pos_below
    tp = positives - pos_below
    fp = negatives - neg_below
    fn = pos_below
    tn = neg_below
    ba = 0.5 * (tp / positives + tn / negatives)
    i = int(np.argmax(ba))  # first maximum => smallest threshold
    tp_i, fp_i, fn_i, tn_i = map(lambda z: int(z[i]), (tp, fp, fn, tn))
    precision = tp_i / (tp_i + fp_i) if tp_i + fp_i else 0.0
    recall = tp_i / positives
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(unique_scores[i]),
        "balanced_accuracy": float(ba[i]),
        "precision": precision, "recall": recall, "f1": f1,
        "TP": tp_i, "TN": tn_i, "FP": fp_i, "FN": fn_i,
        "candidates": int(len(unique_scores)),
        "tie_rule": "smallest score attaining maximum BA",
    }
