"""Initialize steering-switch corrections from constrained quadratic active sets."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from scipy.optimize import minimize
from stable_baselines3 import PPO

from recovery import HERE,HIGH,SPEEDS,XS,common,evaluator,loss,reference_map
from harmonic import make_fit

SCALES=((.88,1.04),(1.12,.96))


def initialize(env,horizon,regularization,steps):
    a,b=env.Ad,env.Bd
    inverse_plant=np.stack([np.linalg.matrix_power(a,j+1) for j in range(horizon)])
    impulse=np.stack([np.linalg.matrix_power(a,j)@b for j in range(horizon)])
    index=np.arange(horizon)
    offset=index[:,None]-index
    lateral=np.where(offset>=0,impulse[offset,0],0.)
    heading=np.where(offset>=0,impulse[offset,2],0.)
    response=np.vstack((lateral,.004*heading))
    free=np.vstack((inverse_plant[:,0],.004*inverse_plant[:,2]))
    hessian=np.linalg.inv(response.T@response+regularization*np.eye(horizon))
    gains=-(hessian@response.T@free)
    predicted=np.ascontiguousarray(reference_map(env))
    p=np.r_[gains[0],make_fit(env)(gains[0])]
    feedforward=p[4:]+predicted@gains[0]
    directions,coefficients,preview=[],[],[]
    for j in steps:
        fraction=hessian[0,j]/hessian[j,j]
        t=.01*j
        transported=feedforward@np.array([[1., 5*t, 12.5*t**2],[0.,1.,5*t],[0.,0.,1.]])
        for sign in (1.,-1.):
            directions.append(gains[j]*XS*sign)
            coefficients.append(-sign*fraction)
            preview.append(sign*transported)
    return p,np.asarray(directions),np.full(8,.5),np.asarray(coefficients),predicted,None,np.asarray(preview)


def read_old(speed):
    r=next(r for r in json.loads((HERE/"polished_fit.json").read_text()) if r["speed"]==speed)
    keys=("weights","directions","thresholds","coefficients","reference_map")
    return (*[np.asarray(r[k]) for k in keys],None,np.zeros((8,3)))


def fit(speed):
    common.torch.set_num_threads(1)
    source=PPO.load(HIGH/"gn_policy_72.zip",device="cpu")
    baseline=read_old(speed)
    sim=evaluator()
    env,_,_=common.configure_env("LANS","dlc",0,speed,1.,1.,"fixed_road")
    data=common.dataset(speed,0,SCALES,range(54630,54664),source,vary_resets=True)
    valid=common.dataset(speed,0,SCALES,range(55314,55339),source,vary_resets=True)
    second=common.dataset(speed,0,((.879,1.042),(1.121,.962)),range(55730,55751),source,vary_resets=True)
    pool=[baseline]
    for steps in ((1,2,3,4),(1,2,4,6),(1,3,5,7)):
        for horizon in (18,31):
            for regularization in (.000045,.000004,.0000004):
                pool.append(initialize(env,horizon,regularization,steps))

    def objective(p,width=None):
        return loss(sim(p,data,steps=width),.8)

    for initial in sorted(pool,key=lambda p:objective(p,24))[:5]:
        p,d,t,c,m,_,q=initial
        max_gain=np.r_[35.,.6,.62,.06]
        wt=np.r_[max_gain,np.minimum(.26,t-.006),np.full(8,.32)]
        def replace(z):
            shift=z[:4]*wt[:4]
            param=np.r_[p[:4]+shift,p[4:]-m@shift]
            return param,d,t+z[4:12]*wt[4:12],c+z[12:]*wt[12:],m,None,q
        def target(v):
            return objective(replace(v),21)
        fitted=minimize(target,np.zeros(len(wt)),method="Powell",bounds=[(-1.,1.)]*len(wt),
                        options={"maxfev":3500,"xtol":1e-6,"ftol":1e-10})
        pool.append(replace(fitted.x))
    print(speed,"analytic candidates scored",flush=True)
    def validation(w):
        return loss(sim(w,valid),.8)+.5*loss(sim(w,second),.8)
    winner=min(pool,key=validation)
    p,d,t,c,m,_,q=winner
    result=dict(speed=speed,weights=p.tolist(),directions=d.tolist(),thresholds=t.tolist(),coefficients=c.tolist(),
                reference_map=m.tolist(),preview_weights=q.tolist(),
                validation=common.stat(sim(winner,valid)),previous_validation=common.stat(sim(baseline,valid)))
    print(result,flush=True)
    return result


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        result=list(pool.map(fit,SPEEDS))
    (HERE/"active_fit.json").write_text(json.dumps(result,indent=2)+"\n")
