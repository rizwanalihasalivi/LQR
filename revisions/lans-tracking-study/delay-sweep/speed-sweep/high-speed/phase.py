"""Locate the nominal error gap by comparing startup error and a steering lower bound."""
import csv
from itertools import product
import json

import numpy as np
from stable_baselines3 import PPO

from fit_high import HERE,common,prepare
from nominal_bound import bound

SPEEDS=(120,150)
PREFIX=80


def main():
    common.torch.set_num_threads(1)
    frozen=list(csv.DictReader((HERE.parent/"rank_by_path.csv").open()))
    model=PPO.load(HERE.parent/"low-speed/tail_policy_72.zip",device="cpu")
    scales=common.sweep.P["test"]["scales"]
    seeds=common.sweep.P["test"]["seeds"]
    result=[]
    for speed in SPEEDS:
        env,length,_=common.configure_env("LANS","dlc",0,speed,1.,1.,"fixed_road")
        lags=np.arange(PREFIX)[:,None]-np.arange(PREFIX)[None,:]
        impulse=np.array([(np.linalg.matrix_power(env.Ad,j)@env.Bd)[0] for j in range(PREFIX)])
        response=np.where(lags>=0,impulse[lags],0.)
        pure=[]
        for (amplitude,scale), path, seed in product(scales,common.PATHS,seeds):
            env,_,ref=common.configure_env("LANS",path,0,speed,amplitude,scale,"fixed_road")
            env.reset(seed=seed)
            xi=env.x[0]
            x=env.x.copy()
            trace=[]
            for t in range(PREFIX):
                dd,ddd=ref[2:4,t]
                x=env.Ad@x+(env.Gd*dd+env.Hd*ddd)/env.vx
                trace.append(x[0])
            pure.append(bound(response,np.asarray(trace),xi,length))
        base=common.dataset(speed,0,scales,seeds,model)
        evaluate,_,_=prepare(speed,0)
        combined=json.loads((HERE/"nominal_combined.json").read_text())
        p=np.array(next(r["weights"] for r in combined if r["speed"]==speed))
        early=evaluate(p,base,steps=PREFIX)[:,0]
        entire=evaluate(p,base)[:,0]
        indexes=np.arange(len(pure))//len(seeds)%4
        for i,path in enumerate(common.PATHS):
            mask=indexes==i
            peer=next(float(r["rmse_penalized_m"]) for r in frozen if
                      (r["profile"],int(r["speed_kmh"]),r["delay_ms"],r["controller"],r["path"])
                      ==("fixed_road",speed,"0","MPC",path))
            record=dict(speed_kmh=speed,path=path,mpc_rmse_m=peer,lower_bound_m=np.array(pure)[mask].mean(),
                        actor_early_80_rmse_m=early[mask].mean(),actor_full_rmse_m=entire[mask].mean())
            print(record,flush=True)
            result.append(record)
    with (HERE/"nominal_error_phases.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=result[0],lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)


if __name__=="__main__":
    main()
