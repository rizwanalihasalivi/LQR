"""Polish bounded steering timing with deterministic closed-loop training."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from scipy.optimize import minimize
from stable_baselines3 import PPO

from recovery import HERE,HIGH,SPEEDS,common,evaluator,loss

TRAIN=((.875,1.038),(1.12,.958),(.94,1.025),(1.06,.973))
CHECK=((.878,1.044),(1.123,.958),(.95,1.016),(1.04,.989))
VERIFY=((.889,1.037),(1.115,.962),(.98,1.002))


def fit(speed):
    common.torch.set_num_threads(1)
    source=PPO.load(HIGH/"gn_policy_72.zip",device="cpu")
    evaluate=evaluator()
    train=common.dataset(speed,0,TRAIN,range(46720,46740),source,vary_resets=True)
    valid=common.dataset(speed,0,CHECK,range(48010,48025),source,vary_resets=True)
    verify=common.dataset(speed,0,VERIFY,range(49500,49516),source,vary_resets=True)
    settings=json.loads((HERE/"recovery_fit.json").read_text())
    r=next(r for r in settings if r["speed"]==speed)
    settings=tuple(np.array(r[k]) for k in ("weights","directions","thresholds","coefficients","reference_map"))
    options=[settings]
    p,d,t,c,reference=settings
    speed_width=np.r_[85.,2.1,1.5,.15,np.full(len(c),.72),np.minimum(.24,t-.008)]

    def translate(params,origin=settings):
        p,d,t,c,ref=origin
        nx=p.copy()
        change=params[:4]*speed_width[:4]
        nx[:4]+=change
        nx[4:]-=ref@change
        n=len(c)
        coeff=c+params[4:4+n]*speed_width[4:4+n]
        threshold=t+params[4+n:]*speed_width[4+n:]
        return nx,d,threshold,coeff,ref

    def objective(params, origin, prefix):
        return loss(evaluate(translate(params,origin),train,steps=prefix),1.)
    bounds=[(-1.,1.)]*len(speed_width)
    for prefix in (12,28):
        origin=min(options,key=lambda candidate:loss(evaluate(candidate,train,steps=prefix),1.))
        result=minimize(objective,np.zeros(len(bounds)),args=(origin,prefix),bounds=bounds,method="Powell",
                        options={"maxfev":4600,"xtol":1e-5,"ftol":1e-10})
        options.append(translate(result.x,origin))
    origin=min(options,key=lambda candidate:loss(evaluate(candidate,train,steps=45),1.))
    rng=np.random.default_rng(speed+54691)
    for center in (np.zeros(len(bounds)),rng.uniform(-.008,.008,len(bounds))):
        result=minimize(objective,center,args=(origin,36),bounds=bounds,method="Nelder-Mead",
                        options={"maxiter":2500,"xatol":1e-7,"fatol":1e-12})
        options.append(translate(result.x,origin))
    def selection(candidate):
        return sum(loss(evaluate(candidate,group),1.) for group in (valid,verify))
    selected=min(options,key=selection)
    keys=("weights","directions","thresholds","coefficients","reference_map")
    record=dict(speed=speed,**{name:value.tolist() for name,value in zip(keys,selected,strict=True)},
                validation=common.stat(evaluate(selected,valid)),
                previous_validation=common.stat(evaluate(settings,valid)),
                secondary_validation=common.stat(evaluate(selected,verify)))
    print(speed,record["validation"],record["previous_validation"],record["secondary_validation"],flush=True)
    return record


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        result=list(pool.map(fit,SPEEDS))
    (HERE/"polished_fit.json").write_text(json.dumps(result,indent=2)+"\n")
