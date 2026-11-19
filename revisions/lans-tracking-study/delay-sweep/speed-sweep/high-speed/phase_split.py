"""Fit fast startup steering while keeping precise low-error steering on the same actor."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import differential_evolution,minimize
from stable_baselines3 import PPO

from fit_high import HERE,common,prepare,loss
from harmonic import make_fit

SPEEDS=(120,150)
TRAIN=((.867,1.04),(1.131,.96),(.958,1.014),(1.05,.974))
VALID=((.881,1.044),(1.119,.957),(.95,1.012))


def fit(speed):
    evaluate,_,_=prepare(speed,0)
    env,_,_=common.configure_env("LANS","dlc",0,speed,1.,1.,"fixed_road")
    source=PPO.load(HERE.parent/"low-speed/tail_policy_72.zip",device="cpu")
    data=common.dataset(speed,0,TRAIN,range(43810,43825),source,vary_resets=True)
    validation=common.dataset(speed,0,VALID,range(44010,44023),source,vary_resets=True)
    options=[json.loads((HERE/f"nominal_{v}.json").read_text()) for v in ("combined","gn")]
    bases=[np.array(next(r["weights"] for r in opts if r["speed"]==speed)) for opts in options]
    settled=min(bases,key=lambda p:loss(evaluate(p,data),.9))
    compute=make_fit(env)
    a,b=env.Ad,env.Bd

    def make(z):
        feedback=z[:4]
        p=np.r_[feedback,compute(feedback),np.zeros(14)]
        perturb=np.r_[settled-p,z[4]]
        return p, perturb

    def objective(z):
        p,correction=make(z)
        return loss(evaluate(p,data,tail=correction,steps=105),.8)

    starts=[]
    for r in (.02,.006,.001,.00004):
        P=solve_discrete_are(a,b[:,None],np.diag([1000.,0.,.001,0.]),np.array([[r]]))
        feedback=-(b@P@a)/(r+b@P@b)
        for radius in (.0005,.001,.0018,.003,.005):
            starts.append(np.r_[feedback,radius])
    bounds=[(-260.,-25.),(-4.5,.1),(-4.,0.),(-.32,.18),(.00018,.007)]
    lo,hi=np.array(bounds).T
    starts=[np.clip(p,lo,hi) for p in starts]
    peak=min(starts,key=objective)
    rng=np.random.default_rng(1150+speed)
    initial=np.clip(peak+rng.normal(size=(64,5))*[18.,.25,.30,.035,.0007],lo,hi)
    search=differential_evolution(objective,bounds,x0=peak,init=initial,seed=rng,maxiter=105,
                                  polish=False,tol=1e-8)
    starts.append(search.x)
    fine=minimize(objective,search.x,method="Nelder-Mead",bounds=bounds,
                  options={"maxiter":1000,"xatol":1e-8})
    starts.append(fine.x)
    candidates=[(p,None) for p in bases]+[make(z) for z in starts]
    for p,tail in sorted(candidates,key=lambda row:loss(evaluate(row[0],data,tail=row[1]),.65))[:3]:
        if tail is None:
            continue
        def objective_full(shift):
            candidate=p.copy()
            candidate[:7]+=shift
            correction=tail.copy()
            correction[:7]-=shift
            return loss(evaluate(candidate,data,tail=correction),.65)
        widths=np.array([24.,.45,.55,.05,.09,.16,.25])
        q=minimize(objective_full,np.zeros(7),method="Nelder-Mead",bounds=list(zip(-widths,widths)),
                   options={"maxiter":850,"xatol":1e-8})
        p=p.copy()
        tail=tail.copy()
        p[:7]+=q.x
        tail[:7]-=q.x
        candidates.append((p,tail))
    p,tail=min(candidates,key=lambda row:loss(evaluate(row[0],validation,tail=row[1]),.75))
    result=dict(speed=speed,delay=0,weights=p.tolist(),tail=None if tail is None else tail.tolist(),
                validation=common.stat(evaluate(p,validation,tail=tail)),
                prior_validation=common.stat(evaluate(settled,validation)))
    print(result,flush=True)
    return result


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(fit,SPEEDS))
    parent=json.loads((HERE/"nominal_gn.json").read_text())
    rows.extend(r for r in parent if r["speed"] not in SPEEDS)
    (HERE/"nominal_split.json").write_text(json.dumps(rows,indent=2)+"\n")
