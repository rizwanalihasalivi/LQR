from pathlib import Path
import json
import numpy as np
from core import StudyEnv,teacher,KR,beta
from features import whiten,unwhiten,T,NetworkEnv
from batch_sim import prepare,build_cases,rollout
ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'results/teacher_search.json').read_text())['selected']['parameters']
cases=build_cases([22000]);batch,flags=rollout(prepare(cases),p)
scalar=[];passed=[]
for c in cases:
 e=StudyEnv(kind='lans',path=c['path'],delay_mode=c['delay'],mass=c['mass'],amplitude=c['amplitude'],time_scale=c['time_scale']);obs,_=e.reset(seed=c['seed'])
 initial=float(e.x[0]);errors=[]
 for k in range(1000):
  d=min(200,max(obs[9]*200,p['proxy']*obs[10]*200))
  action=[np.clip((teacher(obs,p)+KR@e.x)/beta(d),-1,1)]
  obs,_,t,tr,info=e.step(action);errors.append(info['error'])
  if t or tr:break
 n=len(errors);sse=initial**2+np.square(errors).sum();scalar.append(np.sqrt((sse+(1000-n)*max(2.5,abs(errors[-1]))**2)/1001));passed.append(not t and tr)
difference=float(np.max(abs(batch-np.array(scalar))))
print('MAX DIFFERENCE',difference,'FLAGS MATCH',np.array_equal(flags,passed),flush=True)
for c,a,b in zip(cases,batch,scalar):
 if abs(a-b)>1e-6:print(c['path'],c['delay'],c['mass'],a,b,a-b,flush=True)
# The fourth-derivative estimate amplifies float32 preview rounding. Scalar
# and vectorized arithmetic differ by up to0.2mm here; final models are
# evaluated exclusively in the original scalar simulator. Preserve this limit.
assert difference<5e-4 and np.array_equal(flags,passed)
rng=np.random.default_rng(773);x=rng.normal(size=(50,11))
inverse=float(np.max(abs(unwhiten(whiten(x))-x)))
assert np.linalg.matrix_rank(T)==4 and inverse<1e-3
from stable_baselines3 import PPO
import core
def forbidden(*args,**kwargs):raise RuntimeError('Offline teacher invoked during deployment')
core.teacher=forbidden
model=PPO.load(ROOT/'models/limited_lans_72/policy.zip',device='cpu')
e=NetworkEnv(kind='lans',delay_mode='none');obs,_=e.reset(seed=782)
for k in range(100):
 action=model.predict(obs,deterministic=True)[0];obs,_,t,tr,i=e.step(action)
 if t or tr:break
out=dict(batch_vs_scalar_cases=len(cases),maximum_metric_difference=difference,batch_scalar_strict_1e6_match=bool(difference<1e-6),teacher_roundoff_metric_tolerance_m=.0005,completion_parity=bool(np.array_equal(flags,passed)),feature_transform_rank=4,transform_roundtrip_error=inverse,deployed_network_steps_without_offline_teacher=k+1,features=11,scope='Verifies same plant/queue and existing information. No analytic teacher is deployed as LANS.')
(ROOT/'results/checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
