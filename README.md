# LGMMA-X

**Anomaly Detection and Density-Based Scoring of Historical XAU/USD Intraday Data Using an LSTM Autoencoder and Gaussian Mixture Model**

LGMMA-X is a collaborative undergraduate computer-science research project investigating unusual patterns in historical **five-minute XAU/USD** candlesticks. A sequence-to-sequence LSTM autoencoder learns reconstruction behavior; a Gaussian mixture model (GMM) estimates the density of reconstruction errors. Performance is assessed against explicitly defined, retrospective **price-based proxy labels**, not independently verified market events.

> **Research-only.** This repository does not provide trading signals, predict future prices, or establish the causes of historical market movements.

## Method

```text
5-minute OHLC candles
  → timestamp validation and gap-aware segmentation
  → seven engineered features (log return, candle range/body/wicks,
      12-return rolling volatility, 12-return rolling mean)
  → RobustScaler fitted on training data only
  → overlapping, gap-safe 60 × 7 sequences
  → preserved LSTM autoencoder checkpoint
  → sequence-level mean absolute reconstruction error (MAE)
  → train-fitted GMM, K selected by BIC from {2,3,4,5}
  → anomaly score: −ln p(MAE | GMM)
  → validation-selected balanced-accuracy threshold
  → held-out evaluation against sequence-level proxy labels
```

Raw OHLC values are retained for feature construction and retrospective proxy labeling but are **not fed directly into the LSTM-AE**. Proxy criteria use training-only reference statistics: (1) extreme absolute log return, (2) relative high–low range expansion, (3) 12-return volatility burst, and (4) a two-candle price shock followed by an opposite-direction, four-candle endpoint retracement of at least 50%. A sequence receives a positive proxy label when any constituent candle is flagged. Flash-style labels use future closes for **retrospective annotation**, not real-time detection.

## Verified frozen-inference result — November 23, 2026

The isolated **Phase 5F.2 Airflow DAG** re-executed frozen-checkpoint reconstruction, GMM calibration, and held-out evaluation using a pinned environment. A subsequent source-parity check confirmed the consolidated `src/modeling/verified/` calculation path against the frozen reference.

| Verification | Result |
| --- | ---: |
| Train / validation / test sequences | 101,620 / 21,702 / 21,749 |
| Recomputed MAEs matching exactly | 145,071 of 145,071 |
| Selected mixture components | **5** |
| Validation-derived threshold | **−0.272927805749313** (numerical rounding) |
| Precision / recall / F1 | **96.59% / 79.17% / 87.02%** |
| Balanced accuracy | **83.24%** |
| ROC-AUC / average precision | **91.48% / 98.09%** |
| Confusion matrix (TP, TN, FP, FN) | 14,116 / 3,422 / 498 / 3,713 |
| Airflow verification gate | `PASS_INFERENCE_ONLY` |
| Source-parity computation | `PASS_FULL_SOURCE_PARITY` |

The source-parity computation printed a passing result, but its attempt to **save the JSON report from inside Docker failed because the target was read-only**. A locally saved report was reconstructed from that terminal output; it is not a separately authenticated machine-written output. The original Airflow verification reports provide independent, machine-written evidence for the frozen replay.

**Scope and limitations:** This verifies inference using a **preserved trained checkpoint**, not reproduction of original training from initialization. The original training provenance, cross-environment replication, and dependence-aware confirmatory statistical inference remain unresolved.

### Archived manuscript track is different

A prior manuscript result reports **threshold 0.7242016196**, **F1 88.22%**, and **balanced accuracy 85.58%**. These are **not** the result of the verified November 23 inference pipeline. The discrepancy must remain explicit and must not be resolved by replacing numbers without establishing provenance.

## Baselines and notebooks

The project compares ARIMA–GARCH, Z-score, Isolation Forest, One-Class SVM, LSTM-AE fixed-threshold, and LSTM-AE raw-error-ranking configurations. The two LSTM-AE baselines reuse the same trained autoencoder and isolate the contribution of the GMM scoring step. High anomaly-ranking metrics do not automatically imply better classification, or real-event identification.

The canonical notebooks are:

- `src/notebooks/01_comparative_model_analysis.ipynb` — comparative metrics and proxy-condition analysis.
- `src/notebooks/02_statistical_comparison.ipynb` — exploratory paired comparisons across 10 chronological blocks.

Notebook 02's November 23 status is `EXPLORATORY_NOT_CONFIRMATORY` with **32 comparisons and 18 Holm-adjusted rejections**. The blocks are derived from overlapping sequences in one historical dataset, so results do **not** establish temporal independence or confirmatory general superiority. Fixed-MAE threshold provenance remains provisional.

## Source layout

```text
src/
├── README.md
├── pipeline/                    # Historical data ingestion and preprocessing
├── modeling/
│   ├── architecture/            # Byte-identical, checkpoint-compatible architecture
│   ├── verified/                # Import-safe reconstruction, GMM, calibration, evaluation
│   ├── evaluation/              # Proxy-label criteria and historical evaluation
│   └── ...                      # Historical training/sequence entrypoints
├── baseline/                    # Six comparative approaches
├── notebooks/                   # Canonical Phase 5G analyses
├── airflow/
│   └── phase5f2/
│       ├── Dockerfile
│       ├── compose.yaml
│       └── verified_dags/
└── utils/
```

The Phase 5F.2 DAG calls `lgmma_verified_runner.py` in the repository root, which in turn requires `audit_lgmma_phase5d.py` and a consistent local set of frozen artifacts. Legacy training and evaluation entrypoints are retained for **historical provenance**, not advertised as the verified reproduction path.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_proxy_contracts.py" -v
python -m unittest discover -s tests -p "test_verified_source.py" -v
```

The unit tests validate functional contracts; they are **not** a substitute for frozen-artifact parity. To inspect the isolated Airflow configuration:

```powershell
docker compose -f .\src\airflow\phase5f2\compose.yaml config --quiet
```

A fresh clone **cannot execute the frozen inference replay from code alone**: it needs the matching checkpoint, sequence arrays, scaler, GMM artifacts, proxy labels, manifests, and their hashes. These large or potentially restricted files are deliberately excluded from Git. Do not disable integrity checks or mix outputs from different experiment versions. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Credits and research use

This is collaborative thesis work. Credit the research team and manuscript when reusing the implementation. No investment or trading suitability is established by the reported proxy-label performance.
