from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
def run(seed):
 with (ROOT/'results'/f'evaluate_{seed}.log').open('w') as f:subprocess.run([sys.executable,str(ROOT/'code/evaluate.py'),'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE EVALUATION',seed,flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,[42,43,44]))
subprocess.run([sys.executable,str(ROOT/'code/aggregate.py')],env=env,check=True)
subprocess.run([sys.executable,str(ROOT/'code/timing.py')],env=env,check=True)
