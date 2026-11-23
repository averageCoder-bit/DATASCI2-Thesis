"""Verification-first wrapper around the audited Phase 5D experiment.

Non-destructive: original and reference files are read-only; every execution writes
only within reproduction/verified_dag_runs/<safe run-id>/.
Usage:
  python lgmma_verified_runner.py preflight --run-id manual_001
  python lgmma_verified_runner.py replay --run-id manual_001
  python lgmma_verified_runner.py verify --run-id manual_001

The Phase 5D module is imported unchanged, with only its OUT destination redirected.
This procedure verifies inference/evaluation, not original Exp04 training provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / 'reproduction' / 'verified_dag_runs'
REPLAY_SOURCE = ROOT / 'audit_lgmma_phase5d.py'
MANIFEST = ROOT / 'reproduction' / 'phase5f' / 'artifact_manifest_candidate.json'
P1 = ROOT / 'reproduction' / 'reconstruction_verification.json'
P2 = ROOT / 'reproduction' / 'gmm_calibration' / 'gmm_calibration_verification.json'
P3 = ROOT / 'reproduction' / 'heldout_evaluation' / 'heldout_evaluation_verification.json'

# Source hashes based on original Phase 5F inventory, not inferred from a filename.
PROTECTED = {
    'checkpoint_exp04': ('src/modeling/data/models/lstm_autoencoder_exp04.keras', '013cceafec71c9a0af14bc73d3f52c75a1b6856f804c8a69ae711a579123a785'),
    'recovered_scaler': ('recovered_artifacts/robust_scaler.joblib', '089d3228a984a8bfb7c2fbda37d6c4c981d3e0bbbfc5ad27bf587d4dc2ed3bba'),
    'phase1_train_mae': ('reproduction/train_reconstruction_errors.npy', '1b8c24f8a5d9c8764569b4aa68bb0bdc17b04db775d91a344d760e323995e6d6'),
    'phase1_validation_mae': ('reproduction/validation_reconstruction_errors.npy', '92f51c6ee340b749cb6525bd8bf51768a77b96ce0be1fd106c84392ceb52bd72'),
    'phase1_test_mae': ('reproduction/test_reconstruction_errors.npy', '6184101ff0a55d259c6082543f1a875806a504363ef0980bc433e6c371f50177'),
    'selected_gmm': ('reproduction/gmm_calibration/selected_gmm.joblib', '2ab917ce005c5a7603b8d7d97d40f2e6364d709e4bee48eebad87c5b0dee25e3'),
    'selected_threshold': ('reproduction/gmm_calibration/selected_threshold.txt', '532dd182b3253f13af4a9743b391c734991db58969f67ea27d6b94d63ce04270'),
    'architecture_source': ('src/modeling/architecture/autoencoder.py', 'ffbb34e036856c44ccb78c738420008106c46cbd9f05d8f0f6250b2b877b6773'),
}
# Input array hashes from the verified Phase 1 report; actual checks use the report.
EXPECTED_COUNTS = {'train': 101620, 'validation': 21702, 'test': 21749}
EXPECTED_METRICS = {
    'precision': 0.96592308745039,
    'recall': 0.7917437882102193,
    'f1': 0.8702031254816139,
    'balanced_accuracy': 0.8323514859418444,
    'roc_auc': 0.9147543385518434,
    'average_precision': 0.9809109857622743,
    'TP': 14116, 'TN': 3422, 'FP': 498, 'FN': 3713,
}


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(p: Path) -> dict:
    if not p.is_file():
        raise FileNotFoundError(f'Required verified artifact missing: {p}')
    return json.loads(p.read_text(encoding='utf-8'))


def run_dir(run_id: str) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,160}', run_id) or run_id in ('.', '..'):
        raise ValueError('Run ID must contain only letters, digits, underscores, periods, and hyphens')
    return RUNS_ROOT / run_id


def verify_environment() -> dict:
    import numpy
    import sklearn
    import tensorflow as tf
    actual = {
        'python': '.'.join(map(str, sys.version_info[:3])),
        'numpy': numpy.__version__,
        'scikit_learn': sklearn.__version__,
        'tensorflow': tf.__version__,
        'keras': tf.keras.__version__,
    }
    # Explicitly fail rather than silently obtain numerically different results.
    required = {'numpy': '2.3.2', 'scikit_learn': '1.8.0',
                'tensorflow': '2.20.0', 'keras': '3.11.3'}
    differences = {k: {'required': v, 'actual': actual[k]} for k, v in required.items() if actual[k] != v}
    if differences:
        raise RuntimeError(f'Validated environment mismatch; use matching isolated runtime: {differences}')
    return actual


def input_fingerprints() -> dict:
    expected_manifest = read_json(MANIFEST)
    if expected_manifest.get('status') != 'CANDIDATE_ONLY_NOT_APPROVED_FOR_RELEASE':
        raise RuntimeError('Unexpected inventory manifest status')
    collected = {}
    for name, (relative, expected) in PROTECTED.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f'{name}: {path}')
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f'{name} SHA256 mismatch ({actual})')
        listed = expected_manifest.get('files', {}).get(name)
        if not listed or listed.get('sha256') != actual:
            raise RuntimeError(f'{name} differs from Phase 5F candidate inventory')
        collected[relative] = actual
    p1 = read_json(P1)
    read_json(P2)
    read_json(P3)
    for part in EXPECTED_COUNTS:
        path = ROOT / 'src' / 'modeling' / 'data' / 'sequences' / f'X_{part}.npy'
        expected = p1['partitions'][part]['input_sha256']
        if digest(path) != expected:
            raise RuntimeError(f'{part} sequence input hash mismatch')
        collected[str(path.relative_to(ROOT))] = expected
    for path in (P1, P2, P3, REPLAY_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(path)
        collected[str(path.relative_to(ROOT))] = digest(path)
    return collected


def preflight(folder: Path) -> None:
    # Do all checks before creating output directory to avoid partial runs on failure.
    env = verify_environment()
    hashes = input_fingerprints()
    if folder.exists():
        raise FileExistsError(f'Refusing to reuse existing run folder: {folder}')
    folder.mkdir(parents=True, exist_ok=False)
    payload = {'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'environment': env, 'input_sha256': hashes,
               'note': 'Validated-inference replay only; not training or new hyperparameter search'}
    (folder / 'preflight.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print('PASS: environment and frozen input hashes verified:', folder, flush=True)


def ensure_preflight(folder: Path) -> dict:
    pre = read_json(folder / 'preflight.json')
    if pre.get('status') != 'PASS':
        raise RuntimeError('Preflight did not pass')
    actual_env = verify_environment()
    if actual_env != pre['environment']:
        raise RuntimeError('Environment changed since preflight')
    actual_hashes = input_fingerprints()
    if actual_hashes != pre['input_sha256']:
        raise RuntimeError('Frozen inputs changed since preflight')
    return pre


def replay(folder: Path) -> None:
    ensure_preflight(folder)
    target = folder / 'replay'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite prior replay: {target}')
    spec = importlib.util.spec_from_file_location('lgmma_phase5d_verified_source', REPLAY_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load exact Phase 5D source')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = target  # sole redirection: all read-only inputs stay as in Phase 5D
    module.main()  # already raises on failure, writes FAIL report if computation fails
    report = read_json(target / 'phase5d_verification.json')
    if report.get('status') != 'PASS':
        raise RuntimeError(f'Phase 5D full replay did not pass: {report.get("status")}')
    print('PASS: full verified replay completed', flush=True)


def verify(folder: Path) -> None:
    ensure_preflight(folder)
    report = read_json(folder / 'replay' / 'phase5d_verification.json')
    if report.get('status') != 'PASS':
        raise RuntimeError('Replayed experiment failed or incomplete')
    for part, count in EXPECTED_COUNTS.items():
        r = report['neural_replay'][part]
        if not (r['pass'] and r['hash_identical'] and r['sequences'] == count
                and r['exact_mae_count'] == count and r['max_abs_mae_difference'] == 0.0):
            raise RuntimeError(f'{part} reconstruction did not match exactly')
        path = folder / 'replay' / f'{part}_reconstruction_errors.npy'
        if digest(path) != r['phase1_sha256']:
            raise RuntimeError(f'{part} on-disk replay checksum changed')
    if report['gmm_selection']['selected_k'] != 5 or not report['gmm_selection']['pass']:
        raise RuntimeError('GMM selection mismatch')
    cal = report['threshold_calibration']
    if not cal['pass'] or abs(cal['threshold'] - (-0.27292780574931297)) > 1e-7:
        raise RuntimeError('Validation threshold mismatch')
    heldout = report['heldout_evaluation']
    if not heldout['pass'] or heldout['test_used_for_fit_or_calibration'] is not False:
        raise RuntimeError('Test leakage/replay status failure')
    for key, expected in EXPECTED_METRICS.items():
        actual = heldout['metrics'][key]
        tol = 0 if isinstance(expected, int) else 1e-9
        if abs(actual - expected) > tol:
            raise RuntimeError(f'Unexpected {key}: {actual} != {expected}')
    # Reject if a prior verification exists, never grant freeze approval automatically.
    result = folder / 'verification_gate.json'
    if result.exists():
        raise FileExistsError(f'Verification gate already exists: {result}')
    result.write_text(json.dumps({'status': 'PASS_INFERENCE_ONLY',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'phase5d_report_sha256': digest(folder / 'replay' / 'phase5d_verification.json'),
        'metrics': heldout['metrics'],
        'caveats': ['Original training provenance remains unresolved',
                    'No separate-environment reproducibility established',
                    'Do not automatically promote candidate inventory to final release',
                    'Dependence-aware inferential significance remains unresolved']}, indent=2) + '\n', encoding='utf-8')
    print('PASS: final inference verification gate; candidate still not approved for release', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'replay', 'verify'))
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    folder = run_dir(args.run_id)
    {'preflight': preflight, 'replay': replay, 'verify': verify}[args.action](folder)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'FAIL {type(e).__name__}: {e}', file=sys.stderr, flush=True)
        sys.exit(1)
