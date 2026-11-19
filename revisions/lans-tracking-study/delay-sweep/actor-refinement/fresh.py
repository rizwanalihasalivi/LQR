from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
import itertools
import csv
import sys

import torch

from candidate import CandidateEnv, ORIGINAL_ENV, TARGET, TRAIN_SEEDS, sweep

HERE = Path(__file__).resolve().parent

def run(job):
    label = job[1]
    kind = job[2]
    sweep.NetworkEnv = CandidateEnv if label == 'LANS' else ORIGINAL_ENV
    return sweep.run(job)

if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sweep.P['test'] = {**sweep.P['test'],'seeds': list(range(28710,28715)),
                       'scales': [[.9,1.025],[1.1,.975]]}
    sweep.OUT = HERE
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for seed in TRAIN_SEEDS:
        baseline = sweep.load('limited_ppo',seed)
        sweep.POLICIES['limited_ppo',seed] = baseline
        if mode != 'baselines':
            sweep.POLICIES['limited_lans',seed] = type(baseline).load(HERE/f'ppo_{seed}.zip',device='cpu')
    configs = [('LQR','lqr',None,-1),('MPC','mpc',None,-1)]
    configs.extend((label,kind,name,seed) for (label,kind,name),seed in itertools.product(
        [('LANS','lans','limited_lans'),('PPO','ppo','limited_ppo')],TRAIN_SEEDS))
    if mode == 'baselines':
        configs = [c for c in configs if c[0] != 'LANS']
    elif mode == 'candidate':
        configs = [c for c in configs if c[0] == 'LANS']
    jobs = [(d,*config) for d in TARGET for config in configs]
    episodes = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for batch in pool.map(run,jobs):
            episodes.extend(batch)
            print(batch[0]['delay_ms'],batch[0]['controller'],batch[0]['train_seed'],flush=True)
    if mode == 'baselines':
        assert len(episodes) == 800
        sweep.write_csv('fresh_baselines.csv',episodes)
        sys.exit(0)
    if mode == 'candidate':
        with (HERE/'fresh_baselines.csv').open() as f:
            for row in csv.DictReader(f):
                for key in ['delay_ms','train_seed','physical_seed','steps','completed']:
                    row[key] = int(row[key])
                for key in ['amplitude','time_scale','rmse_observed_m','rmse_penalized_m']:
                    row[key] = float(row[key])
                episodes.append(row)
    assert len(episodes)==1280
    sweep.write_csv('fresh_episodes.csv',episodes)
    summary = []
    for d in TARGET:
        groups = {label:[r for r in episodes if r['delay_ms']==d and r['controller']==label]
                  for label in ['LQR','MPC','LANS','PPO']}
        rows = [dict(delay_ms=d,controller=label,
                     rmse_penalized_m=mean(r['rmse_penalized_m'] for r in group),
                     completed=sum(r['completed'] for r in group),episodes=len(group))
                for label,group in groups.items()]
        for rank,row in enumerate(sorted(rows,key=lambda r:r['rmse_penalized_m']),1):
            summary.append(dict(rank=rank,**row))
    sweep.write_csv('fresh_summary.csv',summary)
