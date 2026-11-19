"""Use a shared, state-dependent steering correction with two spare actor units."""
import json

import numpy as np
from scipy.optimize import differential_evolution,minimize
from stable_baselines3 import PPO

from fit_high import HERE,common,prepare,loss

SPEEDS=(120,150)
TRAIN=((.865,1.04),(1.13,.96),(.955,1.018),(1.06,.97))
CHECK=((.879,1.043),(1.117,.958),(.98,1.012))


def tail_from(values):
    return np.r_[values[:4],np.zeros(17),-values[4]]


def main():
    common.torch.set_num_threads(1)
    model=PPO.load(HERE.parent/"low-speed/tail_policy_72.zip",device="cpu")
    originals=[json.loads((HERE/f"nominal_{n}.json").read_text()) for n in ("combined","final","gn")]
    probes=[]
    parameters=[]
    for i,speed in enumerate(SPEEDS):
        sim,_,_=prepare(speed,0)
        train=common.dataset(speed,0,TRAIN,range(42460,42466),model,vary_resets=True)
        valid=common.dataset(speed,0,CHECK,range(42510,42520),model,vary_resets=True)
        starts=[np.array(next(r["weights"] for r in trial if r["speed"]==speed)) for trial in originals]
        def evaluate(p,d,extra=None,prefix=None,sim=sim):
            return loss(sim(p,d,tail=extra,steps=prefix),.6)
        selected=min(starts,key=lambda p:evaluate(p,train))
        parameters.append(selected.copy())
        probes.append((sim,train,valid,evaluate))
    baseline=[probe[3](p,probe[1],prefix=80) for p,probe in zip(parameters,probes,strict=True)]
    offsets=np.array([[-180.,-3.,-4.,-.15,.0005],
                      [ 80., 3., 4.,.15,.38]])
    lo,hi=offsets
    bounds=list(zip(lo,hi))
    def objective(candidate):
        values=[]
        for p,(sim,train,valid,ev),ref in zip(parameters,probes,baseline,strict=True):
            values.append(ev(p,train,tail_from(candidate),prefix=80)/ref)
        return float(np.mean(values)+.25*max(values))
    candidates=[np.array([0.,0.,0.,0.,.1])]
    for additional in (-25.,-60.,-100.,-150.,40.):
        for threshold in (.02,.08,.15,.25):
            candidates.append(np.array([additional,0.,0.,0.,threshold]))
    start=min(candidates,key=objective)
    rng=np.random.default_rng(94121)
    step=np.array([16.,.45,.42,.025,.025])
    points=np.clip(start+rng.normal(size=(64,5))*step,lo,hi)
    fit=differential_evolution(objective,bounds,x0=start,init=points,maxiter=140,
                               seed=rng,tol=1e-7,polish=False)
    candidates.append(fit.x)
    refine=minimize(objective,fit.x,method="Nelder-Mead",bounds=bounds,
                    options={"maxiter":950,"xatol":1e-7})
    candidates.append(refine.x)
    options=[]
    for candidate in sorted(candidates,key=objective)[:5]:
        extra=tail_from(candidate)
        fitted=[]
        for source,(sim,train,valid,ev) in zip(parameters,probes,strict=True):
            def target(earlier):
                return ev(earlier,train,extra)
            # Preserve the feedforward initialization while allowing recovery to share the new hinge.
            widths=np.r_[32.,1.1,1.4,.10,.18,.8,.7,np.full(14,.9)]
            box=list(zip(source-widths,source+widths))
            solved=minimize(target,source,method="Nelder-Mead",bounds=box,
                            options={"maxiter":1050,"xatol":1e-7})
            fitted.append(min((solved.x,source),key=lambda p:ev(p,valid,extra)))
        scores=[pr[3](p,pr[2],extra) for p,pr in zip(fitted,probes,strict=True)]
        options.append((sum(scores),candidate,fitted))
    _,chosen,fitted=min(options,key=lambda v:v[0])
    extra=tail_from(chosen)
    rows=[]
    for speed,p,source,(sim,train,valid,ev) in zip(SPEEDS,fitted,parameters,probes,strict=True):
        row=dict(speed=speed,delay=0,weights=p.tolist(),validation=common.stat(sim(p,valid,tail=extra)),
                 previous_validation=common.stat(sim(source,valid)))
        print(row,flush=True)
        rows.append(row)
    rows.extend(r for r in originals[-1] if r["speed"] not in SPEEDS)
    (HERE/"nominal_hinge.json").write_text(json.dumps(rows,indent=2)+"\n")
    (HERE/"shared_hinge.json").write_text(json.dumps(dict(direction=chosen[:4].tolist(),
        threshold=float(chosen[4])),indent=2)+"\n")


if __name__=="__main__":
    main()
