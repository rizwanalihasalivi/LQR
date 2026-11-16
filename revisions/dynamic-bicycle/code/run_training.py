from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,sys,os,time,json
ROOT=Path(__file__).resolve().parents[1]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
def run(seed):
 for kind in ['full','no_reward','ppo']:
  log=ROOT/'results'/f'train_{kind}_{seed}.log'
  print('START',seed,kind,flush=True)
  with log.open('w') as f:subprocess.run([sys.executable,str(ROOT/'code/train.py'),'--seed',str(seed),'--kind',kind,'--steps','200704'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',seed,kind,log.read_text().strip(),flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,[42,43,44]))
