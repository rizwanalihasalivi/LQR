"""Test the archived reset response with a separately fitted steady-state correction."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from scipy.optimize import minimize
from stable_baselines3 import PPO

from fit_high import HERE,ORIGINAL,common,prepare,loss

SPEEDS=(120,150)
TRAIN=((.875,1.039),(1.12,.958),(.943,1.025),(1.07,.973))
VALID=((.886,1.041),(1.126,.961),(.98,1.006))


def fit(speed):
    sim,_,_=prepare(speed,0)
    model=PPO.load(HERE.parent/"low-speed/tail_policy_72.zip",device="cpu")
    training=common.dataset(speed,0,TRAIN,range(44104,44114),model,vary_resets=True)
    validation=common.dataset(speed,0,VALID,range(44200,44212),model,vary_resets=True)
    base=np.array(next(r["weights"] for r in json.loads((HERE/"nominal_gn.json").read_text()) if r["speed"]==speed))
    alternative=np.array(next(r["weights"] for r in json.loads((HERE/"nominal_final.json").read_text()) if r["speed"]==speed))
    opts=[(base,None)]
    for near in (base,alternative):
        for frac in (0.,.25,.5,.75,1.):
            p=ORIGINAL*frac+base*(1-frac)
            for radius in (.00045,.0009,.0015,.0028,.0045,.006,.009,.012):
                opts.append((p,np.r_[near-p,radius]))
    def objective(option,cases,steps=None):
        return loss(sim(option[0],cases,tail=option[1],steps=steps),.75)
    for start,tail in sorted(opts,key=lambda c:objective(c,training))[:4]:
        if tail is None:
            continue
        steady=start+tail[:21]
        weights=np.r_[32.,.7,.8,.08,.15,.8,.8]
        limits=list(zip(start[:7]-weights,start[:7]+weights))
        def tuning(first):
            p=start.copy()
            p[:7]=first
            return objective((p,np.r_[steady-p,tail[-1]]),training,steps=100)
        fitted=minimize(tuning,start[:7],method="Nelder-Mead",bounds=limits,
                        options={"maxiter":700,"xatol":1e-9})
        p=start.copy()
        p[:7]=fitted.x
        opts.append((p,np.r_[steady-p,tail[-1]]))
    p,t=min(opts,key=lambda c:objective(c,validation))
    row=dict(speed=speed,delay=0,weights=p.tolist(),tail=None if t is None else t.tolist(),
             validation=common.stat(sim(p,validation,tail=t)),
             prior_validation=common.stat(sim(base,validation)))
    print(row,flush=True)
    return row


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(fit,SPEEDS))
    rows.extend(r for r in json.loads((HERE/"nominal_gn.json").read_text()) if r["speed"] not in SPEEDS)
    (HERE/"nominal_legacy.json").write_text(json.dumps(rows,indent=2)+"\n")
