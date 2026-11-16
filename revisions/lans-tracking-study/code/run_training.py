from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,subprocess,os
from protocol import P
ROOT=Path(__file__).resolve().parents[1];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
def run(seed):
 for kind in ['lans','ppo']:
  print('START',kind,seed,flush=True)
  with (ROOT/'results'/f'train_{kind}_{seed}.log').open('w') as f:
   subprocess.run([sys.executable,str(ROOT/'code/train.py'),'--kind',kind,'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',kind,seed,flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,P['training']['policy_seeds']))
