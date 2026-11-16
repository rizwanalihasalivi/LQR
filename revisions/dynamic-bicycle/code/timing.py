import csv,json,os,platform,time,random
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from dynamic_env import DynamicEnv
from mpc import MPC
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1);torch.set_num_interop_threads(1)
pol={k:PPO.load(ROOT/'models'/f'{k}_42'/'policy.zip',device='cpu') for k in ['full','no_reward','ppo']}
kinds=['lqr','full','no_reward','no_gain','no_lqr','ppo','mpc'];envs={k:DynamicEnv(variant={'full':'lans','no_reward':'lans','no_gain':'no_gain','no_lqr':'ppo','ppo':'ppo','lqr':'lqr','mpc':'mpc'}[k],delay_mode='none') for k in kinds}
for e in envs.values():e.reset(seed=3200)
mpc=MPC();rng=np.random.default_rng(817);states=rng.normal(0,[.05,.02,.005,.002],(3100,4))
def call(k,j):
 e=envs[k];e.k=j%990;e.x=states[j];obs=e.obs()
 if k=='mpc':return mpc.action(e)
 if k=='lqr':a=[0.]
 else:a=pol[k if k in pol else 'full'].predict(obs,deterministic=True)[0]
 return e.compose(a,0.)[0]
for j in range(100):
 for k in kinds:call(k,j)
rows=[];rand=random.Random(991)
for j in range(100,3100):
 order=kinds.copy();rand.shuffle(order)
 for k in order:
  t=time.perf_counter_ns();u=call(k,j);dt=time.perf_counter_ns()-t
  assert np.isfinite(u);rows.append(dict(kind=k,sample=j-100,ns=dt))
with (ROOT/'results/timing_raw.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
sumrows=[]
for k in kinds:
 vals=np.array([r['ns'] for r in rows if r['kind']==k])/1000
 sumrows.append(dict(kind=k,samples=len(vals),mean_us=float(vals.mean()),median_us=float(np.median(vals)),p95_us=float(np.quantile(vals,.95)),p99_us=float(np.quantile(vals,.99)),max_us=float(vals.max()),mean_reciprocal_hz=float(1e6/vals.mean())))
(ROOT/'results/timing_summary.json').write_text(json.dumps(sumrows,indent=2))
cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown')
meta=dict(platform=platform.platform(),cpu=cpu,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cpu_count=os.cpu_count(),torch_threads=torch.get_num_threads(),batch_size=1,warmup=100,timed_per_method=3000,scope='observation assembly + deterministic controller decision + fusion/clipping; excludes simulator, transport waiting, sensor sampling, actuation, logging',ordering='random interleaving on identical state sequence',energy_measurement='unavailable: /sys/class/powercap not exposed; no energy saving claim',mpc_optimizer_failures=mpc.failures)
(ROOT/'results/runtime.json').write_text(json.dumps(meta,indent=2));print(json.dumps(sumrows,indent=2))
