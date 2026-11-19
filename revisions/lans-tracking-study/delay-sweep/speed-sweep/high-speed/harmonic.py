"""Fit a nominal feedback and preview response using the discrete plant frequency response."""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
from stable_baselines3 import PPO

from fit_high import HERE,common,prepare,loss

SPEEDS=(120,150)
EXTRA_SCALES=((.872,1.041),(1.13,.957),(.949,1.015),(1.065,.976))


def make_fit(env):
    a,b=env.Ad,env.Bd
    vx=env.vx
    # Training-only approximation of smooth-reference and slalom frequency ranges.
    frequencies=np.r_[.15,.35,.7,1.1,1.6,np.pi*vx/25*np.array([.941,.972,1.,1.027,1.052])]
    preview_h=.01*np.arange(1,5)
    z=np.exp(.01j*frequencies)
    features=((1+1j*frequencies[:,None]*preview_h-np.exp(1j*frequencies[:,None]*preview_h))
              /(10*preview_h**2))@common.TRANSFORM
    weights=np.r_[np.ones(5),np.full(5,1.5)]/frequencies**2
    inputs=features*weights[:,None]
    matrix=np.vstack((inputs.real,inputs.imag))
    force=(-env.Gd*frequencies[:,None]**2-1j*env.Hd*frequencies[:,None]**3)/vx
    eye=np.eye(4)
    def compute(feedback):
        closed=a+np.outer(b,feedback)
        solution=np.linalg.solve(z[:,None,None]*eye-closed,np.stack((force,np.broadcast_to(b,force.shape)),axis=-1))
        expected=-solution[:,0,0]/solution[:,0,1]*weights
        return np.linalg.lstsq(matrix,np.r_[expected.real,expected.imag],rcond=None)[0]
    return compute


def fit(speed):
    evaluate,train,valid=prepare(speed,0)
    env,_,_=common.configure_env("LANS","dlc",0,speed,1.,1.,"fixed_road")
    model=PPO.load(HERE.parent/"low-speed/tail_policy_72.zip",device="cpu")
    extra=common.dataset(speed,0,EXTRA_SCALES,range(41855,41860),model,vary_resets=True)
    coefficients=make_fit(env)
    a,b=env.Ad,env.Bd
    def policy(feedback):
        q=coefficients(feedback)
        return np.r_[feedback,q,np.zeros(14)]
    def objective(p):
        return loss(evaluate(p,train),.8)
    def calibration(p):
        return (loss(evaluate(p,valid),.8)+loss(evaluate(p,extra),.8))/2
    candidates=[]
    for r in (.04,.015,.004,.0003,.00001):
        for heading in (.0005,.005,.04):
            P=solve_discrete_are(a,b[:,None],np.diag([1000.,0.,heading,0.]),np.array([[r]]))
            feedback=-(b@P@a)/(r+b@P@b)
            candidates.append(policy(feedback))
    old=json.loads((HERE/"nominal_combined.json").read_text())
    incumbent=np.array(next(r["weights"] for r in old if r["speed"]==speed))
    candidates.append(incumbent)
    bounds=[(-260.,-25.),(-4.,-.1),(-3.3,-.3),(-.25,.2)]
    for p in sorted(candidates,key=objective)[:4]:
        result=minimize(lambda k:objective(policy(k)),p[:4],method="Nelder-Mead",bounds=bounds,
                        options={"maxiter":600,"xatol":1e-7})
        candidates.append(policy(result.x))
    # Give a calibrated finite-horizon correction room around the physical initialization.
    for p in sorted(candidates,key=objective)[:3]:
        def tune(parameters):
            q=p.copy()
            q[:7]=parameters
            return objective(q)
        widths=np.array([30.,.8,.65,.07,.10,.45,.8])
        domain=list(zip(p[:7]-widths,p[:7]+widths))
        result=minimize(tune,p[:7],bounds=domain,method="Nelder-Mead",
                        options={"maxiter":600,"xatol":1e-7})
        q=p.copy()
        q[:7]=result.x
        candidates.append(q)
    chosen=min(candidates,key=calibration)
    print(speed,common.stat(evaluate(chosen,valid)),common.stat(evaluate(incumbent,valid)),
          common.stat(evaluate(chosen,extra)),flush=True)
    return dict(speed=speed,delay=0,weights=chosen.tolist(),validation=common.stat(evaluate(chosen,valid)),
                previous_validation=common.stat(evaluate(incumbent,valid)),
                secondary_validation=common.stat(evaluate(chosen,extra)))


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(fit,SPEEDS))
    previous=json.loads((HERE/"nominal_combined.json").read_text())
    rows.extend(r for r in previous if r["speed"] not in SPEEDS)
    (HERE/"nominal_final.json").write_text(json.dumps(rows,indent=2)+"\n")
