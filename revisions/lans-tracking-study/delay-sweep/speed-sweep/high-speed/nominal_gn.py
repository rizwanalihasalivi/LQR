"""Refine nominal feedback by propagating steering sensitivities through actual rollout states."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from stable_baselines3 import PPO

from fit_high import HERE,common,prepare,loss

SPEEDS=(120,150)
DT=.01
XS=np.array([.1,.5,.05,.2])
TRAIN=((.869,1.038),(1.13,.96),(.934,1.032),(1.05,.969))
CHECK=((.877,1.041),(1.117,.959),(.957,1.016))
TRAIN_SEEDS=range(42040,42052)
CHECK_SEEDS=range(42140,42150)
BOUNDS=np.r_[105.,3.,3.,.30,1.5,2.,2.,np.full(14,4.)]


def rollout(weights,data,horizon):
    n,a,b,items=data
    force,refs,initial,_=items
    x=initial.copy()
    s=np.zeros((len(x),4,len(weights)))
    errors=np.empty((horizon+1,len(x)))
    slopes=np.zeros((horizon+1,len(x),len(weights)))
    errors[0]=x[:,0]
    kr=common.KR
    hs=DT*np.arange(1,5)
    directions=common.DIRECTIONS/XS[None,:]
    for t in range(horizon):
        reference=refs[:,t]
        vx=reference[:,2:3]
        lateral=(x[:,0]+reference[:,0])[:,None]
        heading=(x[:,2]+reference[:,1])[:,None]
        raw=(lateral+vx*heading*hs-reference[:,3:7]).astype(np.float32).astype(float)
        cleaned=((raw-x[:,:1]-vx*x[:,2:3]*hs)/(10*hs**2)).astype(np.float32)
        q=(cleaned@common.TRANSFORM).astype(np.float32)
        z=(x/XS).astype(np.float32) @ common.DIRECTIONS.T
        positive=np.maximum(z-common.THRESHOLDS,0.)
        negative=np.maximum(-z-common.THRESHOLDS,0.)
        features=np.column_stack((x,q,positive-negative))
        base=-x@kr
        raw_command=features@weights-base
        cmd=(base+raw_command.clip(-1.5,1.5))
        applied=cmd.clip(-.5,.5)
        state_slope=weights[:4]+kr+((positive+negative>0)*weights[7:])@directions
        control_slope=features+np.einsum("ni,nip->np",state_slope,s)
        control_slope*=np.abs(raw_command[:,None])<1.5
        control_slope-=np.einsum("i,nip->np",kr,s)
        control_slope*=np.abs(cmd[:,None])<.5
        s=a@s+b[None,:,None]*control_slope[:,None,:]
        x=x@a.T+b*applied[:,None]+force[:,t]
        errors[t+1]=x[:,0]
        slopes[t+1]=s[:,0]
    rmse=np.sqrt(np.mean(errors**2,axis=0))
    scale=1/np.sqrt(np.maximum(rmse,1e-8))
    residual=(errors*scale[None,:]).ravel()
    jacobian=(slopes*scale[None,:,None]).reshape(-1,len(weights))
    return residual,jacobian


def fit(speed):
    ev,_,_=prepare(speed,0)
    model=PPO.load(HERE.parent/"low-speed/tail_policy_72.zip",device="cpu")
    train=common.dataset(speed,0,TRAIN,TRAIN_SEEDS,model,vary_resets=True)
    validation=common.dataset(speed,0,CHECK,CHECK_SEEDS,model,vary_resets=True)
    pool=[json.loads((HERE/f"nominal_{variant}.json").read_text()) for variant in ("combined","final")]
    starts=[np.array(next(r["weights"] for r in rows if r["speed"]==speed)) for rows in pool]
    def score(p,data,steps=None):
        return loss(ev(p,data,steps=steps),.8)
    candidates=starts.copy()
    for mode,current in enumerate(starts):
        current=current.copy()
        initial=current.copy()
        damping=.003
        for index in range(30):
            horizon=80 if mode==0 and index<16 else train[0]
            r,j=rollout(current,train,horizon)
            h=j.T@j
            step=np.linalg.solve(h+np.diag(damping*np.maximum(h.diagonal(),1e-8)),j.T@r)
            step *= min(1., .20/max(np.max(abs(step)/BOUNDS),1e-12))
            old=score(current,train,steps=horizon)
            accepted=False
            for split in range(7):
                candidate=np.clip(current-step*.5**split,initial-BOUNDS,initial+BOUNDS)
                if score(candidate,train,steps=horizon)<old:
                    current=candidate
                    damping=max(.000001,damping*.75)
                    accepted=True
                    break
            if not accepted:
                damping*=4.
            if index in (7,15,22,29):
                candidates.append(current.copy())
    best=min(candidates,key=lambda p:score(p,validation))
    row=dict(speed=speed,delay=0,weights=best.tolist(),validation=common.stat(ev(best,validation)),
             prior_validation=common.stat(ev(starts[-1],validation)),train=common.stat(ev(best,train)))
    print(row,flush=True)
    return row


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(fit,SPEEDS))
    previous=json.loads((HERE/"nominal_final.json").read_text())
    rows.extend(r for r in previous if r["speed"] not in SPEEDS)
    (HERE/"nominal_gn.json").write_text(json.dumps(rows,indent=2)+"\n")
