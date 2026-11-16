from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys,subprocess,os,json,time
from study_plan import PLAN
ROOT=Path(__file__).resolve().parents[1]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
# Seal all checkpoints and the gain selection before testing fresh cases.
for kind in ['lans','ppo']:
 for seed in PLAN['training_seeds']:
  assert (ROOT/'models'/f'{kind}_{seed}'/'training.json').exists()
marker=ROOT/'results/test_started.json';assert not marker.exists(),'Fresh test is not to be silently rerun.'
marker.write_text(json.dumps(dict(started_at_unix=time.time(),stage='test',plan=PLAN),indent=2))
jobs=[(kind,seed) for kind in ['lans','ppo'] for seed in PLAN['training_seeds']]+[(kind,-1) for kind in ['lqr','fast_lqr','mpc']]
def run(pair):
 kind,seed=pair
 with (ROOT/'results'/f'eval_{kind}_{seed}.log').open('w') as f:
  subprocess.run([sys.executable,str(ROOT/'code/evaluate.py'),'--stage','test','--kind',kind,'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',kind,seed,flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
