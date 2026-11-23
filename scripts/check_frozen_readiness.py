"""Read-only readiness audit for local frozen LGMMA-X Airflow replay.
Does not import the runner or execute model code. Run at repository root:
    python scripts/check_frozen_readiness.py
Exit 0 = required files and literal protected SHA-256 hashes found.
Exit 2 = missing/mismatched inputs. Airflow still needs the verified runtime.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,ast,sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'lgmma_verified_runner.py'
if not SOURCE.is_file():
    print('MISSING',SOURCE);sys.exit(2)
tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
assignments={}
for node in tree.body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        try:assignments[node.targets[0].id]=ast.literal_eval(node.value)
        except (ValueError,TypeError,RecursionError):pass
protected=assignments.get('PROTECTED')
if not isinstance(protected,dict):raise SystemExit('Could not parse runner PROTECTED map')
required=['audit_lgmma_phase5d.py',
          'reproduction/phase5f/artifact_manifest_candidate.json',
          'reproduction/reconstruction_verification.json',
          'reproduction/gmm_calibration/gmm_calibration_verification.json',
          'reproduction/heldout_evaluation/heldout_evaluation_verification.json',
          'src/modeling/data/sequences/X_train.npy',
          'src/modeling/data/sequences/X_validation.npy',
          'src/modeling/data/sequences/X_test.npy']

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()
errors=[]
for name in required:
    path=ROOT/name
    ok=path.is_file()
    print(('PRESENT ' if ok else 'MISSING ')+name)
    if not ok:errors.append(name)
for label,(name,expected) in protected.items():
    path=ROOT/name
    actual=sha256(path) if path.is_file() else None
    ok=actual==expected
    print(('HASH_OK ' if ok else 'HASH_FAIL ')+label+' '+name)
    if not ok:errors.append(name)
print(f'READINESS: {"PASS" if not errors else "BLOCKED"}; missing/mismatched={len(errors)}')
if errors:sys.exit(2)
