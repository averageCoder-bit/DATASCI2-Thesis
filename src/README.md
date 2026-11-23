# LGMMA-X source layout

The reproducible **frozen-inference** reference is the isolated Phase 5F.2
Airflow DAG at `airflow/phase5f2/verified_dags/lgmma_x_verified_reproduction.py`,
which invokes the repository-root `lgmma_verified_runner.py`. The runner and
the root `audit_lgmma_phase5d.py` were **not relocated** by this cleanup.

| Directory | Status and responsibility |
|---|---|
| `modeling/architecture/` | Original checkpoint-compatible TensorFlow model; preserve byte-for-byte |
| `modeling/verified/` | Import-safe, reviewed reconstruction, GMM selection, validation calibration, and evaluation utilities; NO writes on import |
| `modeling/evaluation/proxy_labels.py` | Four training-derived proxy rules; user's separately tested diagnostic correction remains in place |
| `pipeline/` | Original historical ETL; known headerless CSV first-row issue, not certified end-to-end |
| `modeling/{split,scaling,sequence,train}.py` | Original historical preprocessing/training entrypoints; NOT frozen-inference proof |
| `baseline/` | Historical six-baseline implementations; not all independently re-run from scratch under Phase 5F.2 |
| `airflow/phase5f2/` | Isolated verified inference DAG; preserve local `compose.yaml`, Dockerfile and state volumes |
| `notebooks/` | Analysis/notebook copies; reconcile against chosen frozen result track before publication |

## Research result tracks

- **Verified preserved checkpoint, November 23 2026:** K=5,
  validation threshold `-0.27292780574931297`, test N=21,749,
  F1=0.8702031254816139, BA=0.8323514859418444; status
  `PASS_INFERENCE_ONLY`.
- **Archived manuscript results:** threshold `0.7242016196`,
  F1=0.8822, BA=0.8558. **Do not conflate these tracks.**

The new `modeling/verified/` code is a source-consolidation candidate until
`scripts/check_source_parity.py --refit` passes against local preserved
artifacts. Passing the existing Airflow DAG validates its *original* runner,
not necessarily this newly extracted reusable code. This package never
overwrites frozen reference outputs.

## Usage

From repo root (TensorFlow optional for non-inference functions):

```python
from src.modeling.verified.calibration import negative_log_density, select_validation_threshold
from src.modeling.verified.evaluation import align_proxy_labels, evaluate_heldout
```

For frozen-run reproduction, use the established Airflow DAG, **not** the
historical training DAG. Note that any old CLI entrypoints moved into the
separately backed-up legacy archive are no longer supported as active DAG
tasks. Preserve them for method provenance and potential retraining study.
