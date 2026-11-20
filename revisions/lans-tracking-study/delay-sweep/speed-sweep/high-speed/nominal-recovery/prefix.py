"""Bound lateral error with the retained actor's initial commands fixed."""
import csv
from itertools import product
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from stable_baselines3 import PPO
import torch

HERE=Path(__file__).resolve().parent
HIGH=HERE.parent
sys.path.insert(0,str(HIGH))
from fit_high import common
from nominal_bound import bound

SPEEDS=(120,150)
STEPS=55
LOCKS=(0,1,2,3,4,6,8,12,18,25,40)


def infer(matrix,free,initial,length,commands,n):
    forced=free+matrix[:,:n]@commands[:n]
    return bound(matrix[:,n:],forced,initial,length)


def main():
    torch.set_num_threads(1)
    predictor=PPO.load(HIGH/"gn_policy_72.zip",device="cpu")
    frozen=list(csv.DictReader((HIGH.parent/"rank_by_speed_and_delay.csv").open()))
    rows=[]
    for speed in SPEEDS:
        env,length,_=common.configure_env("LANS","dlc",0,speed,1.,1.,"fixed_road")
        impulse=np.array([(np.linalg.matrix_power(env.Ad,t)@env.Bd)[0] for t in range(STEPS)])
        indices=np.arange(STEPS)
        lag=indices[:,None]-indices[None,:]
        matrix=np.where(lag>=0,impulse[lag],0.)
        values={n:[] for n in LOCKS}
        for (amp,scale),path,seed in product(common.sweep.P["test"]["scales"],common.PATHS,
                                             common.sweep.P["test"]["seeds"]):
            env,_,ref=common.configure_env("LANS",path,0,speed,amp,scale,"fixed_road")
            env.reset(seed=seed)
            env.ref=ref
            obs=env.obs()
            state=env.x.copy()
            free=[]
            commands=[]
            initial=state[0]
            for tick in range(STEPS):
                td=ref[2:4,tick]
                state=env.Ad@state+(env.Gd*td[0]+env.Hd*td[1])/env.vx
                free.append(state[0])
                obs,_,failed,_,info=env.step(predictor.predict(obs,deterministic=True)[0])
                assert not failed
                commands.append(info["applied"])
            free=np.asarray(free)
            commands=np.asarray(commands)
            for n in LOCKS:
                values[n].append(infer(matrix,free,initial,length,commands,n))
        mpc=next(float(r["rmse_penalized_m"]) for r in frozen if
                 (r["profile"],int(r["speed_kmh"]),r["delay_ms"],r["controller"])==("fixed_road",speed,"0","MPC"))
        for n,data in values.items():
            rows.append(dict(speed_kmh=speed,fixed_prefix=n,scored_episodes=len(data),
                             mpc_rmse_m=mpc,lans_lower_bound_m=mean(data)))
        print(speed,[(n,round((mean(data)-mpc)*1e6,3)) for n,data in values.items()],flush=True)
    with (HERE/"locked_prefix.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0],lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    main()
