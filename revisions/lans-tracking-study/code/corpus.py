from pathlib import Path
import numpy as np,json
from batch_sim import prepare,rollout
from core import teacher,KR,beta,XS
from features import whiten
ROOT=Path(__file__).resolve().parents[1]
params=json.loads((ROOT/'results/teacher_search.json').read_text())['selected']['parameters']
rng=np.random.default_rng(88000);cases=[]
for j in range(384):
 cases.append(dict(path=str(rng.choice(['dlc','step','sine','slalom'])),delay=str(rng.choice(['none','mixed','fixed200'])),mass=float(rng.uniform(1100,2200)),amplitude=float(rng.uniform(.8,1.2)),time_scale=float(rng.uniform(.85,1.15)),seed=int(rng.integers(88000,98000))))
score,ok,X,Y=rollout(prepare(cases),params,collect=True)
# Existing-state perturbations provide feedback coverage; no new observations.
pick=rng.integers(0,len(X),len(X));aug=X[pick].copy();aug[:,:4]+=rng.normal(0,.35,aug[:,:4].shape)
X=np.concatenate([X,aug]);U=teacher(X,params);U=np.clip(U,-.5,.5)
d=np.minimum(200,np.maximum(X[:,9]*200,params['proxy']*X[:,10]*200))
res=np.clip((U+(X[:,:4]*XS)@KR)/beta(d),-1,1)
Z=whiten(X)
np.savez_compressed(ROOT/'results/corpus.npz',X=Z.astype(np.float32),lans_target=res.astype(np.float32),ppo_target=(U/.5).astype(np.float32))
meta=dict(rows=len(X),rollout_cases=len(cases),teacher_completed=int(ok.sum()),seed=88000,teacher_parameters=params,case_mean_penalized_rmse=float(np.mean(score)),perturbation='Gaussian0.35 on four normalized current-state features; teacher target recomputed from the same11 causal features',scope='Offline training corpus only; no final-test data.')
(ROOT/'results/corpus.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
