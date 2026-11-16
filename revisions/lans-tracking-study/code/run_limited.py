from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,sys,os,json,time
ROOT=Path(__file__).resolve().parents[1]
amendment=dict(reason='Unrestricted fine-tuning regressed from84/84 to74/84 LANS validation completions. Do not discard those outcomes.',change='Keep the same six BC checkpoints; freeze actor hidden layers and logstd; project output-layer parameters within1e-5 physical-action scaled distance of BC weights before every rollout and after learning.',stage='validation-driven corrective trial before final test',primary_lans='limited_lans',all_seeds=[72,73,74],PPO_reference_selection='Select strongest whole PPO reference among limited_ppo,rl_ppo,original_ppo on validation completion then mean penalized RMSE.',time=time.time())
path=ROOT/'results/corrective_trial.json';assert not path.exists()
path.write_text(json.dumps(amendment,indent=2))
previous=ROOT/'results/validation_selection.json'
backup=ROOT/'results/validation_selection_unrestricted.json'
if previous.exists() and not backup.exists():backup.write_bytes(previous.read_bytes())
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
def run(seed):
 for kind in ['lans','ppo']:
  with (ROOT/'results'/f'limited_{kind}_{seed}.log').open('w') as f:subprocess.run([sys.executable,str(ROOT/'code/limited_train.py'),'--kind',kind,'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',kind,seed,flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,[72,73,74]))
