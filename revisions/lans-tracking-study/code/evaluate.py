from pathlib import Path
import sys,argparse,json,csv
import numpy as np
import torch
from stable_baselines3 import PPO
from core import ImprovedEnv,KR
from features import NetworkEnv,unwhiten
from dynamic_env import design,DynamicEnv
from mpc import MPC
from matched_mpc import MatchedMPC
from protocol import P
SUPPORTED_KINDS=('limited_lans','limited_ppo','rl_lans','rl_ppo','bc_lans','bc_ppo','original_ppo','previous_lans','fast_lqr','robust_lqr','mpc','matched_mpc')
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent/'lans-robust-fallback/code'))
from robust_env import RobustEnv
def cases(stage):
 seeds=[22000,22001] if stage=='validation' else P['test']['seeds']
 scales=[[1.,1.]] if stage=='validation' else P['test']['scales']
 for amp,ts in scales:
  for path in ['dlc','step','sine','slalom']:
   for delay in ['none','mixed','fixed200']:
    for ps in seeds:yield dict(path=path,delay=delay,mass=1650.,amplitude=amp,time_scale=ts,physical_seed=ps)
  for mass in [1100.,2200.]:
   for ps in seeds:yield dict(path='dlc',delay='none',mass=mass,amplitude=amp,time_scale=ts,physical_seed=ps)
def load(kind,seed):
 if kind not in SUPPORTED_KINDS:raise ValueError(f'Unsupported controller: {kind}')
 if kind in ['limited_lans','limited_ppo']:
  family=kind.split('_')[1]
  return PPO.load(ROOT/'models'/f'limited_{family}_{seed}'/'policy.zip',device='cpu')
 if kind in ['rl_lans','rl_ppo','bc_lans','bc_ppo']:
  family=kind.split('_')[1];checkpoint='policy' if kind.startswith('rl') else 'bc'
  return PPO.load(ROOT/'models'/f'{family}_{seed}'/(checkpoint+'.zip'),device='cpu')
 if kind=='original_ppo':return PPO.load(ROOT.parent/'dynamic-bicycle/models'/f'ppo_{seed}/policy.zip',device='cpu')
 if kind=='previous_lans':return PPO.load(ROOT.parent/'lans-robust-fallback/models'/f'lans_{seed}/policy.zip',device='cpu')
 return None
def ensure_fresh_output(out,kind,seed,overwrite=False):
 if not overwrite and ((out/f'{kind}_{seed}.csv').exists() or any((out/'traces').glob(f'{kind}_{seed}_*.npz'))):
  raise FileExistsError(f'Results already exist for {kind}, seed {seed} in {out}; use a fresh study or explicit --overwrite.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','test'],required=True);p.add_argument('--kind',choices=SUPPORTED_KINDS,required=True);p.add_argument('--seed',type=int,default=-1);p.add_argument('--overwrite',action='store_true');a=p.parse_args()
 out=ROOT/'results'/a.stage;ensure_fresh_output(out,a.kind,a.seed,a.overwrite)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);policy=load(a.kind,a.seed)
 rows=[];out.mkdir(exist_ok=True);(out/'traces').mkdir(exist_ok=True)
 for c in cases(a.stage):
  kwargs=dict(path=c['path'],mass=c['mass'],delay_mode=c['delay'],amplitude=c['amplitude'],time_scale=c['time_scale'])
  if a.kind=='previous_lans':e=RobustEnv(kind='lans',**kwargs)
  elif a.kind=='original_ppo':e=ImprovedEnv(kind='ppo',residual_scale=.5,beta_decay=.8,**kwargs)
  else:e=NetworkEnv(kind='ppo' if a.kind.endswith('_ppo') else ('lqr' if a.kind in ['fast_lqr','robust_lqr'] else ('mpc' if a.kind in ['mpc','matched_mpc'] else 'lans')),**kwargs)
  if a.kind=='fast_lqr':e.K=design()
  obs,_=e.reset(seed=c['physical_seed']);initial=float(e.x[0]);mpc=MatchedMPC() if a.kind=='matched_mpc' else (MPC() if a.kind=='mpc' else None);trace=[]
  for k in range(1000):
   before=e.x.copy()
   if a.kind=='original_ppo':obs=DynamicEnv.obs(e)
   action=policy.predict(obs,deterministic=True)[0] if policy else ([mpc.action(e)/.5] if mpc else [0.])
   obs,_,t,tr,i=e.step(action)
   trace.append([k*.01,i['error'],i['heading_error'],i['u'],i['applied'],i['residual'],i['gain'],i['delay_ms'],i['queue'],*before])
   if t or tr:break
  arr=np.asarray(trace);n=len(arr);sse=initial**2+np.square(arr[:,1]).sum()
  rows.append(dict(kind=a.kind,train_seed=a.seed,**c,steps=n,completed=not t and tr,rmse_penalized=float(np.sqrt((sse+(1000-n)*max(2.5,abs(arr[-1,1]))**2)/1001)),rmse_observed=float(np.sqrt(sse/(n+1))),optimizer_failures=mpc.failures if mpc else 0))
  key=f"{a.kind}_{a.seed}_{c['physical_seed']}_{c['path']}_{c['delay']}_m{c['mass']:g}_a{c['amplitude']:g}_t{c['time_scale']:g}"
  np.savez_compressed(out/'traces'/f'{key}.npz',trace=arr,initial_error=initial)
 with (out/f'{a.kind}_{a.seed}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(a.stage,a.kind,a.seed,len(rows),sum(r['completed'] for r in rows),np.mean([r['rmse_penalized'] for r in rows]),flush=True)
