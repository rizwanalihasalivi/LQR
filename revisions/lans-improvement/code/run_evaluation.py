from concurrent.futures import ThreadPoolExecutor
import argparse,subprocess,sys,os,json,csv,time,hashlib
from pathlib import Path
import numpy as np
from protocol import P
ROOT=Path(__file__).resolve().parents[1];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','test'],required=True);a=p.parse_args()
if a.stage=='validation':names=list(P['candidates'])
else:
 s=json.loads((ROOT/'results/selection.json').read_text());names=[s['lans'],s['ppo']]
jobs=[(name,seed) for name in names for seed in P['training_seeds']]+[('lqr',-1),('mpc',-1)]
def run(job):
 name,seed=job
 dest=ROOT/'results'/a.stage;dest.mkdir(exist_ok=True)
 with (dest/f'{name}_{seed}.log').open('w') as f:subprocess.run([sys.executable,str(ROOT/'code/evaluate.py'),'--stage',a.stage,'--name',name,'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',a.stage,name,seed,flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
if a.stage=='validation':
 scores={}
 for name in names:
  rows=[]
  for seed in P['training_seeds']:rows+=list(csv.DictReader((ROOT/'results/validation'/f'{name}_{seed}.csv').open()))
  scores[name]=dict(completion=float(np.mean([r['completed']=='True' for r in rows])),mean_penalized_rmse=float(np.mean([float(r['rmse_penalized']) for r in rows])))
 def select(kind):return sorted([n for n,c in P['candidates'].items() if c['kind']==kind],key=lambda n:(-scores[n]['completion'],scores[n]['mean_penalized_rmse'],n))[0]
 output=dict(lans=select('lans'),ppo=select('ppo'),validation_scores=scores,selection_rule=P['selection'],protocol_sha256=hashlib.sha256((ROOT/'results/protocol.json').read_bytes()).hexdigest(),selected_at_unix=time.time())
 path=ROOT/'results/selection.json';assert not path.exists(),'Selection is immutable for the final test.'
 path.write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
