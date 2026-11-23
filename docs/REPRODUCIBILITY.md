# Reproducibility and provenance

## What has been verified

- **2026-11-23 Airflow Phase 5F.2:** `PASS_INFERENCE_ONLY` using preserved Exp04 weights, a hash-verified scaler and input arrays, independently computed reconstruction errors, GMM selection, validation calibration, and held-out testing.
- **2026-11-23 consolidated-source parity:** `PASS_FULL_SOURCE_PARITY` printed by `scripts/check_source_parity.py`, with all **145,071 MAEs exact**, K=5, validation threshold approximately `-0.272927805749313`, and the same test metrics. Its JSON output path was read-only; a local transcription was saved from terminal output. Retain the original Airflow reports and output capture as separate supporting evidence.
- **Phase 5G notebooks:** both canonical comparative and statistical notebooks executed and produced outputs; Notebook 02 reports exploratory chronological-block comparisons, not independent validation.

## Required frozen inputs (not included in the public source repository)

A reproducible frozen run needs an internally consistent snapshot of:

1. Historical XAU/USD M5 source/export, gap segmentation, processed features and chronological split metadata.
2. Frozen training-fitted RobustScaler and seven-feature ordering.
3. `X_train.npy`, `X_validation.npy`, and `X_test.npy` (60 × 7 windows), with corresponding sequence metadata.
4. Preserved Exp04 LSTM autoencoder checkpoint and its expected architecture SHA-256.
5. Frozen reconstruction-error arrays and their hashes (or a verified method to regenerate them).
6. GMM fitting configuration, validation proxy labels, training-derived proxy cutoffs, threshold, and held-out test proxy labels.
7. Verified runner manifests and reference reports.

The source-only repository is **not a self-contained dataset/model release**. Refer to the experiment manifests and any separately authorized data/model distribution before running the reference replay. Never download arbitrary model weights based only on matching filenames.

## Results must remain tied to one experiment track

| Track | GMM threshold | F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| November 23 preserved-checkpoint replay | −0.272927805749313 | 87.02% | 83.24% |
| Archived manuscript result (unreconciled) | 0.7242016196 | 88.22% | 85.58% |

Do not mix the metrics, hashes, thresholds, or models from these tracks. Resolving the discrepancy requires source-level and artifact provenance, **not** simply changing the threshold in a README or notebook.

## Controls

- Train-only transformations and proxy cutoffs must not use validation or test observations.
- The validation threshold must be selected before held-out evaluation.
- Proxy-label criteria are retrospective *operational definitions*, not independent event labels.
- With stride-1 overlapping windows and 60-candle sequence labels, sample independence is not established.
- The flash-style shock/reversal proxy uses later closes and cannot be described as a real-time event alert.
- Pinned Phase 5F.2 environment (observed): TensorFlow `2.20.0`, Keras `3.11.3`, scikit-learn `1.8.0`, NumPy `2.3.2`. GMM inputs must be converted to float64 for fit and scoring parity.
- Locally generated Airflow `state/` (DB, logs, passwords) must not be committed or shared.
- The original training procedure has **not** been independently reproduced from scratch, and the baseline statistical analysis is exploratory.

## Reproduction code entrypoints

- `lgmma_verified_runner.py` — frozen-input preflight, replay, verification gate.
- `audit_lgmma_phase5d.py` — underlying audited Phase 5D replay, called from the runner.
- `src/airflow/phase5f2/verified_dags/lgmma_x_verified_reproduction.py` — verified Airflow DAG.
- `src/modeling/verified/` — import-safe reusable functions.
- `scripts/check_source_parity.py` — checks consolidated source against frozen arrays and model.
- `tests/test_proxy_contracts.py` and `tests/test_verified_source.py` — fast functional tests.

The repository owner should verify the complete import chain and sanitize source/configuration files before staging. Root-level historical audit scripts and legacy DAGs may have hardcoded paths and are not automatically release-grade.
