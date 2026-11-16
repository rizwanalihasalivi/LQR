from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
def run(pair):
 kind,seed=pair
 with (ROOT/'results'/f'legacy_{kind}_{seed}.log').open('w') as f:
  subprocess.run([sys.executable,str(ROOT/'code/legacy_reference.py'),'--kind',kind,'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE legacy',kind,seed,flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,[(kind,seed) for kind in ['full','ppo'] for seed in [42,43,44]]))
