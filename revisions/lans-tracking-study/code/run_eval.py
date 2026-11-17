from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,subprocess,os,sys,time
import csv,numpy as np
from protocol import P
from evaluate import ensure_fresh_output
ROOT=Path(__file__).resolve().parents[1]

def build_jobs(root,stage,only_limited=False,ppo_reference=None):
 if only_limited and stage!='validation':raise ValueError('--only-limited requires validation stage')
 jobs=[(kind,seed) for kind in ['limited_lans','limited_ppo','rl_lans','rl_ppo','bc_lans','bc_ppo'] for seed in [72,73,74]]
 jobs += [('original_ppo',s) for s in [42,43,44]]+[('previous_lans',s) for s in [62,63,64]]+[(k,-1) for k in ['robust_lqr','fast_lqr','mpc','matched_mpc']]
 if stage=='test':
  if ppo_reference not in ['limited_ppo','rl_ppo','original_ppo']:raise ValueError('A supported validation-selected PPO reference is required')
  jobs=[j for j in jobs if not j[0].startswith('rl_') or j[0]==ppo_reference]
 if stage=='validation' and not only_limited:
  jobs=[j for j in jobs if not j[0].startswith('limited_') or (root/'models'/f'{j[0]}_{j[1]}'/'policy.zip').exists()]
 if only_limited:jobs=[j for j in jobs if j[0].startswith('limited_')]
 return jobs

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','test'],required=True);p.add_argument('--only-limited',action='store_true');p.add_argument('--overwrite',action='store_true');a=p.parse_args()
 selected=None
 if a.stage=='test':
  selected=json.loads((ROOT/'results/validation_selection.json').read_text())['ppo_reference']
  mark=ROOT/'results/test_started.json'
  if mark.exists():raise FileExistsError('Test already started; preserve it and use a fresh study directory.')
 jobs=build_jobs(ROOT,a.stage,a.only_limited,selected)
 out=ROOT/'results'/a.stage
 # Check every destination before launching workers or truncating their logs.
 for kind,seed in jobs:ensure_fresh_output(out,kind,seed,a.overwrite)
 if a.stage=='test':
  with mark.open('x') as f:json.dump(dict(time=time.time(),protocol=P),f,indent=2)
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 def run(pair):
  kind,seed=pair;out.mkdir(exist_ok=True)
  command=[sys.executable,str(ROOT/'code/evaluate.py'),'--stage',a.stage,'--kind',kind,'--seed',str(seed)]
  if a.overwrite:command.append('--overwrite')
  with (out/f'{kind}_{seed}.log').open('w') as f:
   subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',a.stage,kind,seed,flush=True)
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
 if a.stage=='validation':
  scores={}
  for kind,seeds in [('limited_ppo',[72,73,74]),('rl_ppo',[72,73,74]),('original_ppo',[42,43,44])]:
   if not all((ROOT/'results/validation'/f'{kind}_{seed}.csv').exists() for seed in seeds):continue
   rr=[]
   for seed in seeds:
    with (ROOT/'results/validation'/f'{kind}_{seed}.csv').open() as f:rr+=list(csv.DictReader(f))
   scores[kind]=dict(completion=np.mean([r['completed']=='True' for r in rr]),rmse=np.mean([float(r['rmse_penalized']) for r in rr]))
  winner=min(scores,key=lambda k:(-scores[k]['completion'],scores[k]['rmse'],k))
  (ROOT/'results/validation_selection.json').write_text(json.dumps(dict(primary_lans='limited_lans' if 'limited_ppo' in scores else 'rl_lans',ppo_reference=winner,scores=scores,rule='One whole PPO reference selected on validation only, never per final-test case. LANS primary always final PPO-trained model; BC only ablation.',time=time.time()),indent=2))
  print('SELECTED PPO BASELINE',winner,scores,flush=True)

if __name__=='__main__':main()
