"""Phase 5D: independently replay frozen Exp04 -> MAE -> GMM -> calibration -> held-out evaluation.

Place this file in the datasci2-thesis repository root and run:
    python .\\audit_lgmma_phase5d.py

Reads existing original and reproduction artifacts but never modifies them.
Writes exclusively to the NEW reproduction/phase5d/ directory and refuses overwrites.

This is an independent *full-inference rerun within the existing environment*, not
proof of reproducibility on another computer/environment, and not LSTM retraining.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent
REP = ROOT / "reproduction"
P1 = REP / "reconstruction_verification.json"
P2 = REP / "gmm_calibration" / "gmm_calibration_verification.json"
P3 = REP / "heldout_evaluation" / "heldout_evaluation_verification.json"
CHECKPOINT = ROOT / "src/modeling/data/models/lstm_autoencoder_exp04.keras"
SEQDIR = ROOT / "src/modeling/data/sequences"
VALID_LABELS = ROOT / "src/modeling/data/evaluation/validation_proxy_labels.csv"
TEST_LABELS = ROOT / "src/modeling/data/evaluation/test_proxy_labels.csv"
OUT = REP / "phase5d"
BATCH = 64
PARTS = ("train", "validation", "test")
SEQS = {p: SEQDIR / f"X_{p}.npy" for p in PARTS}
COMPONENTS = (2, 3, 4, 5)
GMM_ARGS = dict(covariance_type="full", random_state=42, n_init=10)
MAE_ATOL = 1e-6
MAE_RTOL = 1e-6
SCORE_ATOL = 1e-7
METRIC_ATOL = 1e-9

TENSOR_SPECS = {
    "encoder_lstm_kernel": "encoder/lstm/cell/vars/0",
    "encoder_lstm_recurrent": "encoder/lstm/cell/vars/1",
    "encoder_lstm_bias": "encoder/lstm/cell/vars/2",
    "encoder_latent_kernel": "encoder/latent/vars/0",
    "encoder_latent_bias": "encoder/latent/vars/1",
    "decoder_lstm_kernel": "decoder/lstm/cell/vars/0",
    "decoder_lstm_recurrent": "decoder/lstm/cell/vars/1",
    "decoder_lstm_bias": "decoder/lstm/cell/vars/2",
    "decoder_output_kernel": "decoder/output_layer/layer/vars/0",
    "decoder_output_bias": "decoder/output_layer/layer/vars/1",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")


def same_hash(path: Path, expected: str, label: str):
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 differs: actual={actual}, expected={expected}")


def labels_from_csv(path: Path, expected_length: int):
    df = pd.read_csv(path)
    if not {"sequence_id", "proxy_anomaly"}.issubset(df.columns):
        raise ValueError(f"Missing required label columns: {path}")
    if len(df) != expected_length:
        raise ValueError(f"Label length mismatch: {len(df)} != {expected_length}")
    ids = pd.to_numeric(df["sequence_id"], errors="raise").to_numpy()
    if not np.array_equal(ids, np.arange(expected_length)):
        raise ValueError(f"Label sequence IDs are not ordered 0..N-1: {path}")
    y = pd.to_numeric(df["proxy_anomaly"], errors="raise").to_numpy()
    if not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2:
        raise ValueError(f"Expected two binary proxy-label classes in {path}")
    return y.astype(np.int8)


def restored_exp04(sample: np.ndarray):
    sys.path.insert(0, str(ROOT / "src/modeling"))
    from architecture.autoencoder import LSTMAutoencoder

    with zipfile.ZipFile(CHECKPOINT, "r") as archive:
        saved_weight_bytes = archive.read("model.weights.h5")
    with h5py.File(io.BytesIO(saved_weight_bytes), "r") as handle:
        expected = {name: handle[key][()] for name, key in TENSOR_SPECS.items()}

    model = LSTMAutoencoder(hidden_units=128, latent_dim=32, dropout=0.1,
                            sequence_length=60, feature_count=7)
    # IMPORTANT: initialize nested layers with a REAL forward pass before loading.
    _ = model(tf.convert_to_tensor(sample), training=False)
    with tempfile.TemporaryDirectory() as temp:
        filename = Path(temp) / "model.weights.h5"
        filename.write_bytes(saved_weight_bytes)
        model.load_weights(str(filename))

    actual = {
        "encoder_lstm_kernel": model.encoder.lstm.cell.kernel.numpy(),
        "encoder_lstm_recurrent": model.encoder.lstm.cell.recurrent_kernel.numpy(),
        "encoder_lstm_bias": model.encoder.lstm.cell.bias.numpy(),
        "encoder_latent_kernel": model.encoder.latent.kernel.numpy(),
        "encoder_latent_bias": model.encoder.latent.bias.numpy(),
        "decoder_lstm_kernel": model.decoder.lstm.cell.kernel.numpy(),
        "decoder_lstm_recurrent": model.decoder.lstm.cell.recurrent_kernel.numpy(),
        "decoder_lstm_bias": model.decoder.lstm.cell.bias.numpy(),
        "decoder_output_kernel": model.decoder.output_layer.layer.kernel.numpy(),
        "decoder_output_bias": model.decoder.output_layer.layer.bias.numpy(),
    }
    if len(model.weights) != 10 or not all(np.array_equal(expected[k], actual[k]) for k in TENSOR_SPECS):
        raise RuntimeError("Saved checkpoint tensor restoration failed")
    return model


def reconstruct(model, batch: np.ndarray) -> np.ndarray:
    pred = model(tf.convert_to_tensor(batch), training=False).numpy()
    if pred.shape != batch.shape:
        raise RuntimeError(f"Model output shape mismatch: {pred.shape} vs {batch.shape}")
    return np.mean(np.abs(batch - pred), axis=(1, 2)).astype(np.float32)


def choose_threshold(scores: np.ndarray, labels: np.ndarray):
    # Reimplements Phase 2 search: ascending unique score, maximize BA,
    # first/smallest threshold wins a tie. No test scores/labels involved.
    idx = np.argsort(scores, kind="stable")
    s, y = scores[idx], labels[idx]
    positive = int(y.sum())
    negative = len(y) - positive
    if positive == 0 or negative == 0:
        raise ValueError("Cannot select balanced-accuracy threshold with one label class")
    unique, starts = np.unique(s, return_index=True)
    positive_below = np.r_[0, np.cumsum(y, dtype=np.int64)][starts]
    negative_below = starts - positive_below
    ba = 0.5 * ((positive - positive_below)/positive + negative_below/negative)
    ix = int(np.argmax(ba))
    return float(unique[ix]), float(ba[ix]), int(len(unique))


def evaluate(scores, y, threshold):
    pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "predicted_positive": int(pred.sum()), "proxy_positive": int(y.sum()),
    }, pred


def main():
    dependencies = [P1, P2, P3, CHECKPOINT, VALID_LABELS, TEST_LABELS, *SEQS.values()]
    dependencies += [REP / f"{p}_reconstruction_errors.npy" for p in PARTS]
    for p in dependencies:
        require(p)
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 5D directory: {OUT}")
    p1 = json.loads(P1.read_text(encoding="utf-8"))
    p2 = json.loads(P2.read_text(encoding="utf-8"))
    p3 = json.loads(P3.read_text(encoding="utf-8"))
    same_hash(P1, p2["source_reconstruction_manifest_sha256"], "Phase 1 manifest")
    same_hash(P1, p3["phase1_manifest_sha256"], "Phase 1->3 manifest")
    same_hash(P2, p3["phase2_manifest_sha256"], "Phase 2->3 manifest")
    same_hash(CHECKPOINT, p1["checkpoint_sha256"], "Frozen Exp04 checkpoint")
    same_hash(VALID_LABELS, p2["validation_labels_sha256"], "Validation labels")
    same_hash(TEST_LABELS, p3["test_labels_sha256"], "Test labels")
    if p2.get("test_partition_used") is not False or p3.get("test_used_to_fit_model_or_calibrate_threshold") is not False:
        raise RuntimeError("Prior manifests do not certify held-out test exclusion")
    for partition in PARTS:
        same_hash(SEQS[partition], p1["partitions"][partition]["input_sha256"], partition + " input")
        same_hash(REP / f"{partition}_reconstruction_errors.npy",
                  p1["partitions"][partition]["output_sha256"], partition + " Phase 1 MAE")
    print("PASS: inputs, archived reproduction fingerprints, and test isolation", flush=True)

    # All new outputs live exclusively under Phase 5D.
    OUT.mkdir(parents=True, exist_ok=False)
    verification = {"audit": "Phase 5D independent full-run replay", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "IN_PROGRESS", "environment": {"tensorflow": tf.__version__,
                    "keras": tf.keras.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
                    "comparison_tolerances": {"mae_atol": MAE_ATOL, "mae_rtol": MAE_RTOL,
                                              "score_atol": SCORE_ATOL, "metric_atol": METRIC_ATOL},
                    "sources": {"phase1_manifest_sha256": digest(P1),
                                "phase2_manifest_sha256": digest(P2), "phase3_manifest_sha256": digest(P3),
                                "checkpoint_sha256": digest(CHECKPOINT)},
                    "neural_replay": {}, "gmm_selection": {}, "threshold_calibration": {},
                    "heldout_evaluation": {}}

    try:
        test_seq = np.load(SEQS["test"], mmap_mode="r", allow_pickle=False)
        if test_seq.shape[1:] != (60, 7) or test_seq.dtype != np.float32:
            raise ValueError("Unexpected saved test sequence shape/dtype")
        model = restored_exp04(np.asarray(test_seq[:1]))
        print("PASS: all ten saved Exp04 checkpoint tensors restored", flush=True)
        replay = {}
        for part in PARTS:
            sequences = np.load(SEQS[part], mmap_mode="r", allow_pickle=False)
            earlier = np.load(REP / f"{part}_reconstruction_errors.npy", mmap_mode="r", allow_pickle=False)
            count = p1["partitions"][part]["sequences"]
            if sequences.shape != (count, 60, 7) or sequences.dtype != np.float32 or earlier.shape != (count,):
                raise ValueError(f"Unexpected {part} sequence/MAE dimensions")
            new_file = OUT / f"{part}_reconstruction_errors.npy"
            written = np.lib.format.open_memmap(new_file, mode="w+", dtype=np.float32, shape=(count,))
            exact_count = 0
            close_count = 0
            maxdiff = 0.0
            for begin in range(0, count, BATCH):
                end = min(begin + BATCH, count)
                batch = np.asarray(sequences[begin:end])
                result = reconstruct(model, batch)
                previous = np.asarray(earlier[begin:end])
                if not np.isfinite(result).all():
                    raise RuntimeError(f"Nonfinite MAE at {part} [{begin}:{end}]")
                written[begin:end] = result
                exact_count += int(np.count_nonzero(result == previous))
                close_count += int(np.count_nonzero(np.isclose(result, previous, rtol=MAE_RTOL, atol=MAE_ATOL)))
                maxdiff = max(maxdiff, float(np.max(np.abs(result.astype(np.float64) - previous.astype(np.float64)))))
                if end == count or (end // 8192) != (begin // 8192):
                    print(f"{part}: {end:,}/{count:,} neural reconstructions completed", flush=True)
            written.flush()
            del written
            row = {
                "sequences": int(count), "exact_mae_count": exact_count, "within_tolerance_count": close_count,
                "max_abs_mae_difference": maxdiff,
                "new_sha256": digest(new_file),
                "phase1_sha256": p1["partitions"][part]["output_sha256"],
            }
            row["hash_identical"] = row["new_sha256"] == row["phase1_sha256"]
            row["pass"] = close_count == count
            verification["neural_replay"][part] = row
            print(f"{'PASS' if row['pass'] else 'FAIL'} {part}: exact={exact_count:,}/{count:,}, maxdiff={maxdiff:.3g}, byte-hash-equal={row['hash_identical']}", flush=True)
            if not row["pass"]:
                raise RuntimeError(f"Independent full MAE replay differs beyond tolerance on {part}")
        del model
        replay = {p: np.load(OUT / f"{p}_reconstruction_errors.npy", allow_pickle=False).astype(np.float64) for p in PARTS}
        train = replay["train"].reshape(-1, 1)
        selection = []
        models = {}
        for k in COMPONENTS:
            gmm = GaussianMixture(n_components=k, **GMM_ARGS)
            gmm.fit(train)
            row = {"k": k, "bic": float(gmm.bic(train)), "aic": float(gmm.aic(train)),
                   "converged": bool(gmm.converged_), "iterations": int(gmm.n_iter_)}
            if not row["converged"]:
                raise RuntimeError(f"GMM K={k} failed to converge")
            original_entry = next((x for x in p2["gmm_selection_results"] if x["n_components"] == k), None)
            if original_entry is None:
                raise RuntimeError(f"Missing Phase 2 BIC reference for K={k}")
            row["phase2_bic"] = float(original_entry["bic"])
            row["bic_abs_difference"] = abs(row["bic"] - row["phase2_bic"])
            selection.append(row)
            models[k] = gmm
            print(f"GMM K={k}: BIC={row['bic']:.8f}, delta={row['bic_abs_difference']:.3g}", flush=True)
        selected = min(selection, key=lambda r: r["bic"])["k"]
        if selected != p2["selected_k"]:
            raise RuntimeError(f"GMM selection differs: new K={selected}, Phase 2 K={p2['selected_k']}")
        # BIC can vary by rounding/library settings. Compare within strict numeric tolerance.
        if any(row["bic_abs_difference"] > 0.05 for row in selection):
            raise RuntimeError("GMM BIC changed beyond 0.05; investigate environment before accepting")
        verification["gmm_selection"] = {"selected_k": int(selected), "pass": True, "candidates": selection,
                                          "parameters": GMM_ARGS}
        gmm = models[selected]
        val_y = labels_from_csv(VALID_LABELS, len(replay["validation"]))
        val_scores = -gmm.score_samples(replay["validation"].reshape(-1, 1))
        cutoff, ba, n_candidates = choose_threshold(val_scores, val_y)
        old_cutoff = float(p2["validation_calibration"]["threshold"])
        old_ba = float(p2["validation_calibration"]["balanced_accuracy"])
        if abs(cutoff - old_cutoff) > SCORE_ATOL or abs(ba - old_ba) > METRIC_ATOL:
            raise RuntimeError(f"Validation calibration differs: cutoff {cutoff} vs {old_cutoff}; BA {ba} vs {old_ba}")
        if n_candidates != int(p2["validation_calibration"]["candidates"]):
            raise RuntimeError("Number of threshold candidates differs")
        verification["threshold_calibration"] = {"pass": True, "threshold": cutoff, "phase2_threshold": old_cutoff,
                                                  "threshold_difference": cutoff - old_cutoff,
                                                  "validation_balanced_accuracy": ba,
                                                  "candidate_thresholds": n_candidates}
        print(f"PASS: validation threshold {cutoff:.12f}, BA={ba:.10f}", flush=True)

        # Held-out test data used only after model fit and threshold calibration.
        test_y = labels_from_csv(TEST_LABELS, len(replay["test"]))
        test_scores = -gmm.score_samples(replay["test"].reshape(-1, 1))
        metrics, pred = evaluate(test_scores, test_y, cutoff)
        old_metrics = p3["metrics"]
        metric_diff = {name: float(metrics[name] - old_metrics[name]) for name in metrics}
        all_metrics_match = all(abs(delta) <= METRIC_ATOL for delta in metric_diff.values())
        if not all_metrics_match:
            raise RuntimeError(f"Final held-out metrics differ: {metric_diff}")
        verification["heldout_evaluation"] = {"pass": True, "n_test": len(test_y),
                                               "test_used_for_fit_or_calibration": False,
                                               "metrics": metrics, "deviations_from_phase3": metric_diff,
                                               "mean_test_score": float(np.mean(test_scores))}
        np.save(OUT / "test_anomaly_scores.npy", test_scores)
        np.save(OUT / "test_predictions.npy", pred)
        # Persist GMM fit parameters as arrays (not pickled model); compare/report independently.
        np.savez(OUT / "gmm_fit_parameters.npz", weights=gmm.weights_, means=gmm.means_,
                 covariances=gmm.covariances_, precisions_cholesky=gmm.precisions_cholesky_)
        verification["status"] = "PASS"
        (OUT / "phase5d_verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
        print("PASS: full-run replay, GMM selection, validation threshold, and held-out evaluation", flush=True)
        print("Report:", OUT / "phase5d_verification.json", flush=True)
    except Exception as exc:
        verification["status"] = "FAIL"
        verification["error"] = f"{type(exc).__name__}: {exc}"
        (OUT / "phase5d_verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
        print("FAIL: report saved; original and earlier reproduction artifacts unchanged", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
