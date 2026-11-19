import csv
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

OUT = Path(__file__).resolve().parent
ZERO = OUT/'previous'
from candidate import CandidateEnv, ORIGINAL_ENV, sweep

MODELS = {}
PATHS = ['dlc','step','sine','slalom']
BINS = [(0,1),(1,11),(11,31),(31,101),(101,1001)]

def run(args):
    label, seed, group, physical_seeds, scales = args
    model = MODELS.get((label,seed))
    records=[]
    for name, (amp,ts), ps in itertools.product(PATHS,scales,physical_seeds):
        env = (ORIGINAL_ENV if label=='MPC' else CandidateEnv)(
            kind='mpc' if label=='MPC' else 'lans',path=name,
            delay_mode='fixed0',amplitude=amp,time_scale=ts)
        mpc=sweep.MatchedMPC() if label=='MPC' else None
        obs,_=env.reset(seed=ps)
        state=[env.x.copy()]
        actions=[]
        for k in range(1000):
            action = [mpc.action(env)/.5] if mpc else model.predict(obs,deterministic=True)[0]
            obs,_,terminated,truncated,info=env.step(action)
            state.append(env.x.copy())
            actions.append(info['u'])
            if terminated or truncated:
                break
        assert truncated and not terminated
        x=np.array(state)
        row=dict(group=group,controller=label,checkpoint=seed,path=name,amplitude=amp,
                 time_scale=ts,physical_seed=ps,rmse_m=float(np.sqrt(np.mean(x[:,0]**2))),
                 max_abs_lateral_m=float(abs(x[:,0]).max()),
                 steering_saturation_steps=int(np.sum(np.abs(actions)>=.499999)),
                 signed_steering_samples=' '.join(f'{u:.8g}' for u in actions[:20]))
        for a,b in BINS:
            row[f'sse_{a}_{b-1}']=float(np.sum(x[a:b,0]**2))
        records.append(row)
    return records

if __name__=='__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    mode=sys.argv[1]
    seeds=[72,73,74]
    if mode=='diagnose':
        locations={seed:ZERO/f'ppo_{seed}.zip' for seed in seeds}
        configs=[('prior',list(range(28710,28715)),[[.9,1.025],[1.1,.975]]),
                 ('holdout',list(range(31240,31256)),[[.87,1.035],[1.13,.965]])]
    else:
        locations={seed:OUT/f'ppo_{seed}.zip' for seed in seeds}
        configs=[('prior',list(range(28710,28715)),[[.9,1.025],[1.1,.975]]),
                 ('new',list(range(34520,34536)),[[.925,1.015],[1.075,.985]])]
    for seed, file in locations.items():
        MODELS['LANS',seed]=PPO.load(file,device='cpu')
    jobs=[(label,seed,*config) for config in configs
          for label,seed in [('MPC',-1),*[('LANS',s) for s in seeds]]]
    if len(sys.argv)>2 and sys.argv[2]=='baseline':
        jobs=[job for job in jobs if job[0]=='MPC']
    records=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(run,jobs):
            records.extend(r)
            print(r[0]['group'],r[0]['controller'],r[0]['checkpoint'],len(r),flush=True)
    with (OUT/f'{mode}_traces.csv').open('w',newline='') as f:
        fw=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n')
        fw.writeheader();fw.writerows(records)
