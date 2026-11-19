"""Fit a narrow low-error tracking correction at 15 km/h using spare actor units."""
import json

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
from stable_baselines3 import PPO

from fit import HERE, SWEEP, SPEEDS, configure_env, dataset, prepare, stat, torch


def feedforward(env, feedback):
    closed = env.Ad + np.outer(env.Bd,feedback)
    inverse = np.linalg.inv(np.eye(4)-closed)
    effectiveness = inverse@env.Bd
    last, older = np.zeros((2,4))
    result=[]
    for disturbance in (env.Gd/env.vx,env.Hd/env.vx,np.zeros(4)):
        response=inverse@(disturbance-.01*last-.00005*older)
        cmd=-response[0]/effectiveness[0]
        result.append(cmd)
        last,older=response+effectiveness*cmd,last
    return np.array(result)*[20.,100.,500.]


def main():
    torch.set_num_threads(1)
    train_seeds=[40215,40216]
    verify_seeds=range(40310,40319)
    model=PPO.load(SWEEP/"completion-followup/final_policy_72.zip",device="cpu")
    test,_,_=prepare(15,0)
    train=dataset(15,0,((.86,1.042),(1.12,.954)),train_seeds,model,vary_resets=True)
    valid=dataset(15,0,((.88,1.045),(1.103,.968)),verify_seeds,model,vary_resets=True)
    env,_,_=configure_env("LANS","dlc",0,15,1,1,"fixed_road")
    rows=[json.loads((HERE/f"nominal_{name}.json").read_text()) for name in ("polish","precision")]
    starts=[np.array(next(r["weights"] for r in records if r["speed"]==15)) for records in rows]
    a,b=env.Ad,env.Bd
    feedbacks=[]
    for r in (.02,.005,.0003):
        value=solve_discrete_are(a,b[:,None],np.diag([1000.,0.,.003,0.]),np.array([[r]]))
        feedbacks.append(-(b@value@a)/(r+b@value@b))
    candidates=[(p,None) for p in starts]
    for p in starts:
        for feedback in feedbacks:
            ff=feedforward(env,feedback)
            for radius in (.0008,.0013,.0023):
                replacement=np.r_[feedback,ff,np.zeros(14)]
                candidates.append((p,np.r_[replacement-p,radius]))

    def error(candidate, cases):
        return test(candidate[0],cases,tail=candidate[1])[:,0].mean()

    proposals=sorted(candidates,key=lambda c:error(c,train))[:4]
    for p, correction in proposals:
        if correction is None:
            continue
        def fit(coeff):
            updated=correction.copy()
            updated[4:7]=coeff
            return error((p,updated),train)
        result=minimize(fit,correction[4:7],method="Nelder-Mead",
                        options={"maxiter":100,"xatol":2e-6})
        updated=correction.copy()
        updated[4:7]=result.x
        candidates.append((p,updated))
    p,extra=min(candidates,key=lambda c:error(c,valid))
    result=dict(speed=15,delay=0,weights=p.tolist(),validation=stat(test(p,valid,tail=extra)),
                previous=stat(test(starts[-1],valid)),tail=None if extra is None else extra.tolist())
    print(result,flush=True)
    other=[r for r in rows[0] if r["speed"] in SPEEDS and r["speed"]!=15]
    (HERE/"nominal_tail.json").write_text(json.dumps([result,*other],indent=2)+"\n")


if __name__=="__main__":
    main()
