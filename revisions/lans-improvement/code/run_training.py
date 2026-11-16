import sys,subprocess,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from protocol import P
ROOT=Path(__file__).resolve().parents[1]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
def run(seed):
 for name in P['candidates']:
  print('START',name,seed,flush=True)
  with (ROOT/'results'/f'train_{name}_{seed}.log').open('w') as f:
   subprocess.run([sys.executable,str(ROOT/'code/train.py'),'--name',name,'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',name,seed,flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,P['training_seeds']))
