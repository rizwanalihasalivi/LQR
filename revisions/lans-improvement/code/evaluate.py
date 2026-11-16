import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from env import ImprovedEnv
from protocol import P
from mpc import MPC
ROOT=Path(__file__).resolve().parents[1]
def cases(stage):
 seeds=P['validation_seeds'] if stage=='validation' else P['test_seeds']
 scales=[(1.,1.)] if stage=='validation' else [(.9,1.1),(1.1,.9)]
 for amp,ts in scales:
  for path in ['dlc','step','sine','slalom']:
   for delay in ['none','mixed','fixed200']:
    for seed in seeds:yield dict(path=path,delay=delay,mass=1650.,amplitude=amp,time_scale=ts,physical_seed=seed)
  for mass in [1100.,2200.]:
   for seed in seeds:yield dict(path='dlc',delay='none',mass=mass,amplitude=amp,time_scale=ts,physical_seed=seed)
def rollout(name,seed,policy,case,stage):
 c=P['candidates'].get(name,dict(kind=name,residual_scale=.5,beta_decay=.8))
 e=ImprovedEnv(kind=c['kind'],residual_scale=c['residual_scale'],beta_decay=c['beta_decay'],path=case['path'],mass=case['mass'],delay_mode=case['delay'],amplitude=case['amplitude'],time_scale=case['time_scale'])
 obs,_=e.reset(seed=case['physical_seed']);initial=float(e.x[0])
 mpc=MPC() if name=='mpc' else None;trace=[]
 for k in range(1000):
  x_before=e.x.copy()
  action=[mpc.action(e)/.5] if mpc else ([0.] if policy is None else policy.predict(obs,deterministic=True)[0])
  obs,r,t,tr,i=e.step(action)
  trace.append([k*.01,i['error'],i['heading_error'],i['u'],i['applied'],i['residual'],i['gain'],i['delay_ms'],i['queue'],*x_before])
  if t or tr:break
 a=np.array(trace);n=len(a);sse=initial**2+np.square(a[:,1]).sum()
 row=dict(candidate=name,train_seed=seed,**case,steps=n,completed=not t and tr,rmse=float(np.sqrt(sse/(n+1))),rmse_penalized=float(np.sqrt((sse+(1000-n)*max(2.5,abs(a[-1,1]))**2)/1001)),rms_residual=float(np.sqrt(np.mean(a[:,5]**2))),optimizer_failures=mpc.failures if mpc else 0)
 dest=ROOT/'results'/stage/'traces';dest.mkdir(parents=True,exist_ok=True)
 key=f"{name}_{seed}_{case['physical_seed']}_{case['path']}_{case['delay']}_m{case['mass']:g}_a{case['amplitude']:g}_t{case['time_scale']:g}"
 np.savez_compressed(dest/f'{key}.npz',trace=a,initial_error=initial,columns=np.array(['time','error','heading_error','command','applied','raw_residual','gain','delay_ms','queue','pre_ey','pre_eydot','pre_heading','pre_headingdot']))
 return row
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','test'],required=True);p.add_argument('--name',required=True);p.add_argument('--seed',type=int,default=-1);a=p.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 if a.stage=='test':
  selected=json.loads((ROOT/'results/selection.json').read_text())
  assert a.name in ['lqr','mpc',selected['lans'],selected['ppo']]
 policy=None if a.name in ['lqr','mpc'] else PPO.load(ROOT/'models'/f'{a.name}_{a.seed}'/'policy.zip',device='cpu')
 rows=[rollout(a.name,a.seed,policy,c,a.stage) for c in cases(a.stage)]
 out=ROOT/'results'/a.stage;out.mkdir(exist_ok=True)
 with (out/f'{a.name}_{a.seed}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(a.stage,a.name,a.seed,len(rows),sum(r['completed'] for r in rows),float(np.mean([r['rmse_penalized'] for r in rows])),flush=True)
