"""Fit speed-specific residual coefficients using disjoint seeds and the original FIFO."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import differential_evolution, minimize
from stable_baselines3 import PPO

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"low-speed"))
import fit as common
from tail import feedforward

SPEEDS=(120,150,200)
DELAYS=(0,200,1000)
TRAIN_SCALES=((.865,1.043),(1.132,.956),(.94,1.025),(1.06,.973))
VALID_SCALES=((.878,1.047),(1.125,.951),(1.015,1.003))
TRAIN_SEEDS=range(41020,41026)
VALID_SEEDS=range(41120,41126)
ORIGINAL=np.r_[common.FITS[0]["weights"],[h["weight"] for h in common.HINGES]]
SWEEP=HERE.parents[1]


def prepare(speed,delay):
    common.torch.set_num_threads(1)
    model=PPO.load(HERE.parent/"low-speed/tail_policy_72.zip",device="cpu")
    sim=ctypes.CDLL(str(HERE/"rollout.so")).replay
    f=common.FLOAT_ARRAY
    sim.argtypes=[ctypes.c_int]*5+[ctypes.c_double,f,ctypes.c_void_p]+[f]*11+[ctypes.c_void_p]
    sim.restype=None
    train=common.dataset(speed,delay,TRAIN_SCALES,TRAIN_SEEDS,model,vary_resets=True)
    valid=common.dataset(speed,delay,VALID_SCALES,VALID_SEEDS,model,vary_resets=True)
    authority=common.SCALE*np.exp(-common.DECAY*max(0,delay-5))
    def evaluate(p,data,start=None,*,steps=None,tail=None,directions=None,thresholds=None):
        n,a,b,arrays=data
        results=np.empty((len(arrays[2]),3))
        pointer=None if start is None else start.ctypes.data
        correction=None if tail is None else tail.ctypes.data
        basis=common.DIRECTIONS if directions is None else directions
        levels=common.THRESHOLDS if thresholds is None else thresholds
        sim(len(results),n,n if steps is None else steps,delay//10,len(p)-7,authority,
            np.ascontiguousarray(p),pointer,a,b,common.KR,*arrays,common.TRANSFORM,
            basis,levels,results,correction)
        return results
    return evaluate,train,valid


def loss(rows, weight):
    rmse, complete, lengths=rows.T
    return float(rmse.mean()+weight*(1.-complete.mean()))


def nominal(speed):
    evaluate, train, valid=prepare(speed,0)
    env,_,_=common.configure_env("LANS","dlc",0,speed,1.,1.,"fixed_road")
    a,b=env.Ad,env.Bd
    alternatives=[ORIGINAL]
    for r in (.02,.001,.00003):
        for heading in (.001,.015,.12):
            q=np.diag([1000.,0.,heading,0.])
            P=solve_discrete_are(a,b[:,None],q,np.array([[r]]))
            feedback=-(b@P@a)/(r+b@P@b)
            ff=feedforward(env,feedback)
            alternatives.append(np.r_[feedback,ff,np.zeros(14)])
    if speed==200:
        for gain in (-80.,-130.,-200.):
            alternatives.append(np.r_[gain,-1.4,-1.9,-.041,ORIGINAL[4:7],np.zeros(14)])
    width=np.r_[110.,2.8,5.,.3,1.4,2.,2.,np.full(14,3.)]
    bounds=list(zip(ORIGINAL-width,ORIGINAL+width))
    def objective(p):
        return loss(evaluate(p,train),.45)
    initial=sorted(alternatives,key=objective)[:4]
    for guess in initial:
        clipped=np.clip(guess,ORIGINAL-width,ORIGINAL+width)
        refined=minimize(objective,clipped,method="Nelder-Mead",bounds=bounds,
                         options={"maxiter":1050,"xatol":2e-7,"fatol":1e-11})
        alternatives.append(refined.x)
    # Joint fit of the three preview coefficients on the full course.
    for base in sorted(alternatives,key=objective)[:3]:
        def refine(x):
            candidate=base.copy()
            candidate[4:7]=x
            return objective(candidate)
        result=minimize(refine,base[4:7],method="Nelder-Mead",bounds=bounds[4:7],
                        options={"maxiter":150,"xatol":1e-7})
        p=base.copy()
        p[4:7]=result.x
        alternatives.append(p)
    best=min(alternatives,key=lambda p:loss(evaluate(p,valid),.45))
    result=dict(speed=speed,delay=0,weights=best.tolist(),train=common.stat(evaluate(best,train)),
                validation=common.stat(evaluate(best,valid)),source_validation=common.stat(evaluate(ORIGINAL,valid)))
    print(result,flush=True)
    return result


def delayed(task):
    speed,delay=task
    evaluate,train,valid=prepare(speed,delay)
    zero=json.loads((HERE/"nominal.json").read_text())
    startup=np.array(next(r["weights"] for r in zero if r["speed"]==speed))
    original=np.array(common.FITS[delay]["weights"])
    if delay==1000:
        final=json.loads((SWEEP/"completion-followup/final_fit.json").read_text())
        original=np.array(next(r["weights"] for r in final if r["delay"]==delay))
    bounds=np.array([[-1.25,-.7,-4.,-1.2,-.9,-2.8,-3.2],
                     [.28,.6,4.,1.2,.9,2.8,3.2]])
    lo,hi=bounds
    domain=list(zip(lo,hi))
    options=[original,np.r_[-common.KR,0.,0.,0.],
             np.r_[-.25*common.KR,original[4:]],np.zeros(7),np.r_[np.zeros(4),original[4:]]]
    best_source=original.copy()
    weight=.9 if delay==200 else .55
    def objective(p):
        return loss(evaluate(p,train,startup),weight)
    rng=np.random.default_rng(92201+speed+delay)
    for objective_weight in (weight,.1):
        weight=objective_weight
        best=min(options,key=objective)
        radius=np.array([.035,.045,.22,.08,.09,.32,.5])
        points=np.vstack((best+rng.normal(size=(52,7))*radius,rng.uniform(lo,hi,size=(12,7))))
        result=differential_evolution(objective,domain,x0=best,init=np.clip(points,lo,hi),
                                      maxiter=130,tol=1e-6,polish=False,seed=rng)
        options.append(result.x)
        fitted=minimize(objective,result.x,method="Nelder-Mead",bounds=domain,
                         options={"maxiter":900,"xatol":2e-6})
        options.append(fitted.x)
    weight=.6 if delay==200 else .35
    best=min(options,key=lambda p:loss(evaluate(p,valid,startup),weight))
    result=dict(speed=speed,delay=delay,weights=best.tolist(),train=common.stat(evaluate(best,train,startup)),
                validation=common.stat(evaluate(best,valid,startup)),
                source_validation=common.stat(evaluate(best_source,valid)))
    print(result,flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("nominal","delayed"))
    stage=parser.parse_args().stage
    with ProcessPoolExecutor(max_workers=4) as pool:
        if stage=="nominal":
            rows=list(pool.map(nominal,SPEEDS))
        else:
            rows=list(pool.map(delayed,product(SPEEDS,DELAYS[1:])))
    (HERE/f"{stage}.json").write_text(json.dumps(rows,indent=2)+"\n")
