"""Matched reference checks against the previously published stronger checkpoints.
These are supplementary controls; they do not change validation selection."""
from pathlib import Path
import csv,json,argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from robust_env import ImprovedEnv
from dynamic_env import DynamicEnv
from evaluate import cases
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['full','ppo'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
torch.set_num_threads(1);torch.set_num_interop_threads(1)
policy=PPO.load(ROOT.parent/'dynamic-bicycle/models'/f'{a.kind}_{a.seed}'/'policy.zip',device='cpu')
rows=[];out=ROOT/'results/legacy_reference';out.mkdir(exist_ok=True)
for c in cases('test'):
 e=ImprovedEnv(kind='lans' if a.kind=='full' else 'ppo',residual_scale=.5,beta_decay=.8,path=c['path'],mass=c['mass'],delay_mode=c['delay'],amplitude=c['amplitude'],time_scale=c['time_scale'])
 e.reset(seed=c['physical_seed']);initial=float(e.x[0]);trace=[]
 for k in range(1000):
  # Exact original observation API; no inverse scaling approximation.
  old_obs=DynamicEnv.obs(e)
  action=policy.predict(old_obs,deterministic=True)[0];before=e.x.copy()
  _,_,t,tr,i=e.step(action)
  trace.append([k*.01,i['error'],i['heading_error'],i['u'],i['applied'],i['residual'],i['gain'],i['delay_ms'],i['queue'],*before])
  if t or tr:break
 arr=np.array(trace);n=len(arr);sse=initial**2+np.square(arr[:,1]).sum()
 rows.append(dict(candidate='legacy_'+a.kind,train_seed=a.seed,**c,steps=n,completed=not t and tr,rmse=float(np.sqrt(sse/(n+1))),rmse_penalized=float(np.sqrt((sse+(1000-n)*max(2.5,abs(arr[-1,1]))**2)/1001))))
 dest=out/'traces';dest.mkdir(exist_ok=True)
 key=f"{a.kind}_{a.seed}_{c['physical_seed']}_{c['path']}_{c['delay']}_m{c['mass']:g}_a{c['amplitude']:g}_t{c['time_scale']:g}"
 np.savez_compressed(dest/f'{key}.npz',trace=arr,initial_error=initial)
with (out/f'{a.kind}_{a.seed}.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(a.kind,a.seed,len(rows),sum(r['completed'] for r in rows),flush=True)
