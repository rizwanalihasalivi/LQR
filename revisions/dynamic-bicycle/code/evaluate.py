import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from dynamic_env import DynamicEnv,HORIZON
from mpc import MPC
ROOT=Path(__file__).resolve().parents[1]
def write_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rollout(kind,policy,train_seed,physical_seed,path,delay,mass=1650.,scope='full'):
 variant={'full':'lans','no_reward':'lans','no_gain':'no_gain','no_lqr':'ppo','ppo':'ppo','lqr':'lqr','mpc':'mpc'}[kind]
 e=DynamicEnv(variant=variant,path=path,delay_mode=delay,mass=mass,delay_scope=scope);obs,_=e.reset(seed=physical_seed)
 mpc=MPC() if kind=='mpc' else None
 trace=[];initial=float(e.x[0]);begin=time.perf_counter()
 for k in range(HORIZON):
  if policy is None:action=[0.] if mpc is None else [mpc.action(e)/.5]
  else:action=policy.predict(obs,deterministic=True)[0]
  obs,r,t,tr,i=e.step(action)
  trace.append([k*.01,i['error'],i['heading_error'],i['u'],i['applied'],i['residual'],i['gain'],i['delay_ms'],i['saturation'],i['queue'],i['generation']])
  if t or tr:break
 arr=np.array(trace);n=len(arr);sse=initial**2+np.square(arr[:,1]).sum()
 row=dict(kind=kind,train_seed=train_seed,physical_seed=physical_seed,path=path,delay=delay,scope=scope,mass=mass,steps=n,completed=bool(not t and tr),rmse_observed=float(np.sqrt(sse/(n+1))),rmse_penalized=float(np.sqrt((sse+(HORIZON-n)*max(2.5,abs(arr[-1,1]))**2)/(HORIZON+1))),max_error=float(max(abs(initial),abs(arr[:,1]).max())),rms_residual=float(np.sqrt(np.mean(arr[:,5]**2))),rms_applied_residual=float(np.sqrt(np.mean((arr[:,5]*arr[:,6])**2))),saturated_fraction=float(arr[:,8].mean()),mpc_optimizer_failures=0 if mpc is None else mpc.failures,seconds=time.perf_counter()-begin)
 name=f'{kind}_t{train_seed}_p{physical_seed}_{path}_{delay}_{scope}_m{mass:g}'
 np.savez_compressed(ROOT/'results/traces'/f'{name}.npz',trace=arr,columns=np.array(['t','error','heading_error','command','applied','residual','gain','delay_ms','saturation','queue','applied_generation']))
 return row
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);(ROOT/'results/traces').mkdir(exist_ok=True)
 models={k:PPO.load(ROOT/'models'/f'{k}_{a.seed}'/'policy.zip',device='cpu') for k in ['full','no_reward','ppo']} if a.seed!=-1 else {}
 rows=[]
 kinds=['lqr','mpc'] if a.seed==-1 else ['full','no_reward','no_gain','no_lqr','ppo']
 for kind in kinds:
  policy=None if a.seed==-1 else models[kind if kind in models else 'full']
  for path in ['dlc','step','sine','slalom']:
   for delay in ['none','mixed','fixed200']:
    for ps in range(3200,3205):
     rows.append(rollout(kind,policy,a.seed,ps,path,delay))
   print('DONE',a.seed,kind,path,flush=True)
  # Mass and delay-placement controls use DLC, paired physical seeds.
  for mass in [1100.,2200.]:
   for ps in range(3200,3205):rows.append(rollout(kind,policy,a.seed,ps,'dlc','none',mass))
  if kind in ['full','lqr','ppo']:
   for ps in range(3200,3205):rows.append(rollout(kind,policy,a.seed,ps,'dlc','fixed200',scope='residual'))
  write_csv(ROOT/'results'/f'episodes_{a.seed}.csv',rows)
 print('COMPLETED',a.seed,len(rows),flush=True)
