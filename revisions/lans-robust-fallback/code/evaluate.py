import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from robust_env import RobustEnv
from dynamic_env import design
from mpc import MPC
from study_plan import PLAN
ROOT=Path(__file__).resolve().parents[1]
def cases(stage):
 seeds=PLAN['validation_seeds'] if stage=='validation' else PLAN['test_seeds']
 scales=[[1.,1.]] if stage=='validation' else PLAN['test_reference_scales']
 for amp,ts in scales:
  for path in ['dlc','step','sine','slalom']:
   for delay in ['none','mixed','fixed200']:
    for ps in seeds:yield dict(path=path,delay=delay,mass=1650.,amplitude=amp,time_scale=ts,physical_seed=ps)
  for mass in [1100.,2200.]:
   for ps in seeds:yield dict(path='dlc',delay='none',mass=mass,amplitude=amp,time_scale=ts,physical_seed=ps)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--seed',type=int,default=-1);p.add_argument('--stage',choices=['validation','test'],required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 policy=None if a.seed==-1 else PPO.load(ROOT/'models'/f'{a.kind}_{a.seed}'/'policy.zip',device='cpu')
 out=ROOT/'results'/a.stage;out.mkdir(exist_ok=True);(out/'traces').mkdir(exist_ok=True);rows=[]
 for case in cases(a.stage):
  e=RobustEnv(kind='lqr' if a.kind=='fast_lqr' else a.kind,path=case['path'],mass=case['mass'],delay_mode=case['delay'],amplitude=case['amplitude'],time_scale=case['time_scale'])
  if a.kind=='fast_lqr':e.K=design()
  obs,_=e.reset(seed=case['physical_seed']);initial=float(e.x[0]);mpc=MPC() if a.kind=='mpc' else None;trace=[]
  for k in range(1000):
   before=e.x.copy()
   action=policy.predict(obs,deterministic=True)[0] if policy else ([mpc.action(e)/.5] if mpc else [0.])
   obs,_,t,tr,i=e.step(action)
   trace.append([k*.01,i['error'],i['heading_error'],i['u'],i['applied'],i['residual'],i['gain'],i['delay_ms'],i['queue'],*before])
   if t or tr:break
  arr=np.asarray(trace);n=len(arr);sse=initial**2+np.square(arr[:,1]).sum()
  rows.append(dict(kind=a.kind,train_seed=a.seed,**case,steps=n,completed=bool(not t and tr),rmse=float(np.sqrt(sse/(n+1))),rmse_penalized=float(np.sqrt((sse+(1000-n)*max(2.5,abs(arr[-1,1]))**2)/1001)),optimizer_failures=mpc.failures if mpc else 0))
  key=f"{a.kind}_{a.seed}_{case['physical_seed']}_{case['path']}_{case['delay']}_m{case['mass']:g}_a{case['amplitude']:g}_t{case['time_scale']:g}"
  np.savez_compressed(out/'traces'/f'{key}.npz',trace=arr,initial_error=initial,columns=np.array(['time','error','heading_error','command','applied','residual','gain','delay_ms','queue','pre_ey','pre_eydot','pre_heading','pre_headingdot']))
 with (out/f'{a.kind}_{a.seed}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(a.stage,a.kind,a.seed,len(rows),sum(r['completed'] for r in rows),float(np.mean([r['rmse_penalized'] for r in rows])),flush=True)
