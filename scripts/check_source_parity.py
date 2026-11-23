"""Read-only check of new modeling/verified APIs against frozen November 23 DAG output.

Example (in pinned Phase 5F.2 container, from repo root):
  /opt/lgmma-verified-python/bin/python scripts/check_source_parity.py \
      --run-id 20261123T024650 --refit

--refit actually exercises the new GMM selection source on the same verified
training MAEs; without --refit this is ONLY a score/evaluation parity check.
Writes one new JSON report only if --report is supplied; never overwrites.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling.verified.calibration import (fit_gmm_candidates,negative_log_density,
                                               select_validation_threshold)
from src.modeling.verified.evaluation import align_proxy_labels, evaluate_heldout
from src.modeling.verified.integrity import require_sha256
from src.modeling.verified.config import (CHECKPOINT_EXP04_SHA256,
    PROTECTED_ARCHITECTURE_SHA256,VERIFIED_THRESHOLD,EXPECTED_COUNTS)


def require(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen artifact: {path}")
    return path


def load_mae(part: str, p1: dict):
    path = require(ROOT/'reproduction'/f'{part}_reconstruction_errors.npy')
    require_sha256(path,p1['partitions'][part]['output_sha256'])
    a = np.load(path,allow_pickle=False)
    if a.shape != (EXPECTED_COUNTS[part],) or not np.isfinite(a).all():
        raise ValueError(f'{part} MAE shape or values mismatched')
    return a


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--refit',action='store_true',help='Refit K=2..5 GMM using new source')
    parser.add_argument('--reconstruction',choices=['none','sample','full'],default='none',
                        help='Verify newly extracted model-restoration/inference source')
    parser.add_argument('--report',type=Path,help='Optional NEW JSON report; refuses overwrite')
    args=parser.parse_args()

    run=ROOT/'reproduction'/'verified_dag_runs'/args.run_id
    gate=json.loads(require(run/'verification_gate.json').read_text(encoding='utf8'))
    ref=json.loads(require(run/'replay/phase5d_verification.json').read_text(encoding='utf8'))
    p1=json.loads(require(ROOT/'reproduction/reconstruction_verification.json').read_text(encoding='utf8'))
    p2=json.loads(require(ROOT/'reproduction/gmm_calibration/gmm_calibration_verification.json').read_text(encoding='utf8'))
    if gate.get('status')!='PASS_INFERENCE_ONLY' or ref.get('status')!='PASS':
        raise RuntimeError('Reference Airflow run did not pass')
    require_sha256(ROOT/'src/modeling/architecture/autoencoder.py',PROTECTED_ARCHITECTURE_SHA256)
    require_sha256(ROOT/'src/modeling/data/models/lstm_autoencoder_exp04.keras',CHECKPOINT_EXP04_SHA256)
    train=load_mae('train',p1)
    val=load_mae('validation',p1)
    test=load_mae('test',p1)

    saved_gmm=require(ROOT/'reproduction/gmm_calibration/selected_gmm.joblib')
    require_sha256(saved_gmm,p2['selected_gmm_sha256'])
    gmm=joblib.load(saved_gmm)  # Only load one's own trusted artifact
    if gmm.n_components != 5:
        raise RuntimeError('Frozen GMM must have K=5')
    score_val=negative_log_density(gmm,val)
    score_test=negative_log_density(gmm,test)
    labels_val=pd.read_csv(require(ROOT/'src/modeling/data/evaluation/validation_proxy_labels.csv'))
    labels_test=pd.read_csv(require(ROOT/'src/modeling/data/evaluation/test_proxy_labels.csv'))
    # Verified Phase 2 requires sequential 0..N-1 validation IDs.
    ids_val=np.arange(len(val))
    if not np.array_equal(labels_val.sequence_id.to_numpy(),ids_val):
        raise RuntimeError('Validation sequence IDs reordered relative to verified calibration')
    ids_test=pd.read_csv(require(ROOT/'src/modeling/data/gmm_results/test_anomaly_scores.csv'),
                         usecols=['sequence_id']).sequence_id.to_numpy()
    if not np.array_equal(ids_test,np.arange(len(test))):
        raise RuntimeError('Held-out sequence IDs reordered relative to frozen test input')
    yval=align_proxy_labels(labels_val,ids_val)
    ytest=align_proxy_labels(labels_test,ids_test)
    result_val=select_validation_threshold(score_val,yval)
    selected=float(result_val['threshold'])
    if abs(selected-VERIFIED_THRESHOLD)>1e-7 or abs(selected-ref['threshold_calibration']['threshold'])>1e-7:
        raise RuntimeError(f'New validation threshold differs: {selected}')
    results=evaluate_heldout(score_test,ytest,selected)
    for name,expected in gate['metrics'].items():
        if name not in results:
            continue
        delta=abs(results[name]-expected)
        if delta>(0 if isinstance(expected,int) else 1e-9):
            raise RuntimeError(f'New source differs on {name}: {results[name]} vs {expected}')

    fit_info={'performed':False}
    if args.refit:
        new_gmm, rows=fit_gmm_candidates(train)
        if new_gmm.n_components != ref['gmm_selection']['selected_k']:
            raise RuntimeError('New GMM/BIC selected different K')
        for computed, prior in zip(rows, ref['gmm_selection']['candidates']):
            if computed['n_components']!=prior['k'] or abs(computed['bic']-prior['bic'])>1e-6:
                raise RuntimeError(f'GMM BIC mismatch at K={prior["k"]}')
        newly_scored_val=negative_log_density(new_gmm,val)
        new_t=select_validation_threshold(newly_scored_val,yval)['threshold']
        if abs(new_t-selected)>1e-7:
            raise RuntimeError(f'Refit GMM produced different threshold: {new_t} != {selected}')
        newly_scored_test=negative_log_density(new_gmm,test)
        new_metrics=evaluate_heldout(newly_scored_test,ytest,new_t)
        for name,expected in gate['metrics'].items():
            if name in new_metrics and abs(new_metrics[name]-expected)>(0 if isinstance(expected,int) else 1e-9):
                raise RuntimeError(f'Refit GMM changed heldout metric {name}')
        fit_info={'performed':True,'selected_k':new_gmm.n_components,'BIC':rows,
                  'selected_threshold':new_t}
    reconstruction_info={'mode':args.reconstruction,'performed':False}
    if args.reconstruction!='none':
        from src.modeling.verified.reconstruction import (restored_exp04_model,
                                                          infer_reconstruction_mae)
        sequence_dir=ROOT/'src/modeling/data/sequences'
        checkpoint=ROOT/'src/modeling/data/models/lstm_autoencoder_exp04.keras'
        test_x=np.load(require(sequence_dir/'X_test.npy'),mmap_mode='r',allow_pickle=False)
        # Model initialization/weight verification uses a single test sequence.
        model=restored_exp04_model(checkpoint,np.asarray(test_x[:1]))
        partitions=('train','validation','test') if args.reconstruction=='full' else ('test',)
        recon_results={}
        for part in partitions:
            x=(test_x if part=='test' else np.load(require(sequence_dir/f'X_{part}.npy'),
                                                  mmap_mode='r',allow_pickle=False))
            expected={'train':train,'validation':val,'test':test}[part]
            count=len(x) if args.reconstruction=='full' else min(128,len(x))
            if len(x)!=EXPECTED_COUNTS[part] or x.shape[1:]!=(60,7):
                raise RuntimeError(f'{part} sequence input shape/count changed')
            produced=infer_reconstruction_mae(model,x[:count],batch_size=64)
            delta=np.abs(produced.astype(np.float64)-expected[:count].astype(np.float64))
            max_delta=float(delta.max())
            n_exact=int(np.count_nonzero(produced==expected[:count]))
            if max_delta>1e-6:
                raise RuntimeError(f'New {part} reconstruction differs by {max_delta}')
            if args.reconstruction=='full' and n_exact != count:
                raise RuntimeError(f'Full {part} MAEs are not bitwise identical: {n_exact}/{count}')
            recon_results[part]={'count':int(count),'exact_maes':n_exact,
                                 'max_abs_difference':max_delta}
            print(f'RECON PARITY {part}: {n_exact}/{count} exact, max difference {max_delta:.3g}',flush=True)
        reconstruction_info={'mode':args.reconstruction,'performed':True,
                             'by_partition':recon_results}

    if args.refit and args.reconstruction=='full':
        status='PASS_FULL_SOURCE_PARITY'
    elif args.refit:
        status='PASS_SOURCE_CALIBRATION_PARITY'
    elif args.reconstruction!='none':
        status='PASS_RECONSTRUCTION_CHECK_ONLY'
    else:
        status='PASS_READ_ONLY_SCORE_PARITY'
    payload={'status':status,'created_utc':datetime.now(timezone.utc).isoformat(),
             'verified_run_id':args.run_id,'selected_threshold':selected,
             'test_sequences':len(test),'metrics':results,'refit_gmm':fit_info,
             'reconstruction':reconstruction_info,
             'caveat':'Verified preserved-checkpoint inference only; training/provenance unresolved'}
    print(json.dumps(payload,indent=2))
    if args.report:
        dest=args.report if args.report.is_absolute() else ROOT/args.report
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x',encoding='utf8') as stream:
            json.dump(payload,stream,indent=2)
            stream.write('\n')
        print(f'Saved new parity report: {dest}')

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print(f'FAIL {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(1)
