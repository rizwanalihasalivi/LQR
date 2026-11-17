"""Offline supervision with the same causal inputs; no final cases are used."""
from pathlib import Path
import json
import argparse
import numpy as np
from precise_features import PrecisionEnv  # activates sibling imports
from batch_sim import prepare, rollout
from core import teacher, KR, beta, XS
from features import whiten

ROOT = Path(__file__).resolve().parents[1]
protocol = json.loads((ROOT / 'results/protocol.json').read_text())
parser = argparse.ArgumentParser()
parser.add_argument('--audit-only',action='store_true',help='Replay teacher outcomes without replacing training data')
args = parser.parse_args()
if (ROOT / 'results/corpus.npz').exists() and not args.audit_only:
    raise FileExistsError('Keep the existing corpus; use a fresh campaign.')
rng = np.random.default_rng(protocol['corpus_seed'])
cases = [dict(path=str(rng.choice(['dlc','step','sine','slalom'])),
              delay=str(rng.choice(['none','mixed','fixed200'])),
              mass=float(rng.uniform(1100,2200)),
              amplitude=float(rng.uniform(.8,1.2)),
              time_scale=float(rng.uniform(.85,1.15)),
              seed=int(rng.integers(99000,199000))) for _ in range(384)]
params = protocol['teacher']
score, ok, X, _ = rollout(prepare(cases), params, collect=True)
audit_path = ROOT / 'results/teacher_cases.json'
if audit_path.exists():
    raise FileExistsError('Teacher case audit already exists; preserve it.')
audit_path.write_text(json.dumps([dict(**case,rmse_penalized=float(s),completed=bool(o))
                                 for case,s,o in zip(cases,score,ok)],indent=2))
if args.audit_only:
    previous = json.loads((ROOT/'results/corpus.json').read_text())
    assert previous['teacher_completed'] == int(ok.sum())
    assert abs(previous['mean_penalized_rmse']-float(score.mean())) < 1e-12
    print('Teacher audit replay matches retained corpus metadata; corpus unchanged.')
    raise SystemExit(0)
aug = X[rng.integers(0,len(X),len(X))].copy()
aug[:,:4] += rng.normal(0,.35,aug[:,:4].shape)
X = np.concatenate([X,aug])
U = np.clip(teacher(X,params),-.5,.5)
d = np.minimum(200,np.maximum(X[:,9]*200,params['proxy']*X[:,10]*200))
residual = np.clip((U+(X[:,:4]*XS)@KR)/beta(d),-1,1)
np.savez_compressed(ROOT/'results/corpus.npz',X=whiten(X).astype(np.float32),
                    lans_target=residual.astype(np.float32),ppo_target=(U/.5).astype(np.float32))
meta = dict(rows=len(X),rollout_cases=len(cases),teacher_completed=int(ok.sum()),
            mean_penalized_rmse=float(score.mean()),seed=protocol['corpus_seed'],
            precision_before_float32=True,teacher_is_offline_only=True)
(ROOT/'results/corpus.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta),flush=True)
