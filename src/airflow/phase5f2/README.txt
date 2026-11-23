LGMMA-X Phase 5F.2: isolated Airflow reproducibility candidate
===============================================================
IMPORTANT: This is a candidate configuration, NOT an executed or passed Airflow test.
The first full replay on Linux can fail byte-identical comparison even at the same
package versions due to differences in CPU kernels/platform. Never relax hashes
or numeric gates silently, and never overwrite original artifacts.

Preparation from Windows PowerShell in thesis repository:
  cd "$HOME\Downloads\datasci2-thesis"
  New-Item -ItemType Directory -Force '.\reproduction\verified_dag_runs' | Out-Null
  # Extract this ZIP under .\src\airflow\phase5f2\ (compose.yaml must be in that folder)
  cd '.\src\airflow\phase5f2'
  docker compose -f compose.yaml config
  docker compose -f compose.yaml build
  docker compose -f compose.yaml up -d
  docker compose -f compose.yaml ps

Verify exact numerical interpreter without running experiment:
  docker compose -f compose.yaml exec airflow-verified /opt/lgmma-verified-python/bin/python -c "import sys,numpy,sklearn,tensorflow as tf,keras; print(sys.version.split()[0],numpy.__version__,sklearn.__version__,tf.__version__,keras.__version__)"

Check new instance's web UI at http://localhost:8082 (different from historical :8080).
Only trigger the manually scheduled lgmma_x_verified_reproduction DAG in :8082.
It must run preflight -> replay -> verification gate in that sequence.

After trigger:
  Check .\reproduction\verified_dag_runs\<Airflow ts_nodash>\verification_gate.json
  Expected success status is PASS_INFERENCE_ONLY, not approved for release.

Cleanup the disposable service (preserves ./state and repo reports):
  docker compose -f compose.yaml down
DO NOT use down -v or docker system prune.

If the preflight fails, do not trigger a new run without diagnosing the cause.
If the replay fails on Linux, preserve the failure evidence and do not
misrepresent local Windows reproducibility as cross-platform reproducibility.
The Docker image installs uv managed Python 3.13.7 and numeric packages;
Airflow retains its original Python from apache/airflow:3.3.1.
