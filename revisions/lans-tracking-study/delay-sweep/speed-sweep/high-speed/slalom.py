"""Fit a bounded high-curvature recovery branch using existing preview channels."""
import json

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize,differential_evolution
from stable_baselines3 import PPO

from fit_high import HERE,prepare,common,loss
from tail import feedforward

DIVISORS=np.array([5.,6.])
TRAIN_SCALES=((.86,1.043),(1.132,.954),(.95,1.024),(1.07,.974))
CHECK_SCALES=((.875,1.046),(1.121,.958),(1.018,1.001))
TRAIN_SEEDS=range(41600,41606)
CHECK_SEEDS=range(41640,41646)


def with_path(dataset,scales,seeds,path):
    by_path=np.arange(len(scales)*4*len(seeds))
    index=common.PATHS.index(path)
    keep=(by_path//len(seeds))%4==index
    n,a,b,arrays=dataset
    return n,a,b,tuple(np.ascontiguousarray(arr[keep]) for arr in arrays)


def activation_gap(scales):
    slalom_min=np.inf
    usual_max=-np.inf
    for amp,scale in scales:
        for path in common.PATHS:
            env,n,ref=common.configure_env("LANS",path,0,200,amp,scale,"fixed_road")
            env.ref=ref
            h=.01*np.arange(1,5)
            ticks=np.arange(n)
            raw=ref[0,:n,None]+ref[1,:n,None]*h-ref[0,ticks[:,None]+np.arange(1,5)]
            features=(raw/(10*h*h))@common.TRANSFORM
            level=np.sum(abs(features[:,:2])/DIVISORS,axis=1)
            if path=="slalom":
                slalom_min=min(slalom_min,float(level.min()))
            else:
                usual_max=max(usual_max,float(level.max()))
    if not usual_max<slalom_min:
        raise RuntimeError("the preview features do not separate the sampled curvature ranges")
    return usual_max,slalom_min


def main():
    common.torch.set_num_threads(1)
    model=PPO.load(HERE.parent/"low-speed/tail_policy_72.zip",device="cpu")
    sim,_,_=prepare(200,0)
    train=with_path(common.dataset(200,0,TRAIN_SCALES,TRAIN_SEEDS,model,vary_resets=True),
                    TRAIN_SCALES,TRAIN_SEEDS,"slalom")
    valid=with_path(common.dataset(200,0,CHECK_SCALES,CHECK_SEEDS,model,vary_resets=True),
                    CHECK_SCALES,CHECK_SEEDS,"slalom")
    env,_,_=common.configure_env("LANS","slalom",0,200,1.,1.,"fixed_road")
    a,b=env.Ad,env.Bd
    candidates=[]
    for r in (1.,40.,1000.,5000.):
        state_weights=np.diag([60.,1.,4.,.03])
        P=solve_discrete_are(a,b[:,None],state_weights,np.array([[r]]))
        feedback=-(b@P@a)/(r+b@P@b)
        ff=feedforward(env,feedback)
        for fraction in (0.,.2,.4,.55,.7):
            for omega in (6.7,7.14,7.55):
                co=ff*fraction
                co[0]+=(1-fraction)*(-20*feedback[0]/omega**2+20*feedback[3]/env.vx)
                co[1]-=(1-fraction)*100*(feedback[1]+feedback[2]/env.vx)/omega**2
                candidates.append(np.r_[feedback,co])
    def objective(p):
        return loss(sim(p,train),1.2)
    bounds=[(-3.,.2),(-1.2,.5),(-6.,3.),(-1.5,.5),(-3.,3.),(-4.,4.),(-6.,6.)]
    lo,hi=np.array(bounds).T
    candidates=[np.clip(p,lo,hi) for p in candidates]
    start=min(candidates,key=objective)
    radius=np.array([.14,.04,.22,.04,.10,.10,.16])
    rng=np.random.default_rng(54621)
    initial=np.clip(start+rng.normal(size=(72,7))*radius,lo,hi)
    result=differential_evolution(objective,bounds,x0=start,init=initial,seed=rng,maxiter=170,
                                  polish=False,tol=1e-7)
    candidates.append(result.x)
    for variant in (start,result.x):
        p=minimize(objective,variant,method="Nelder-Mead",bounds=bounds,
                   options={"maxiter":1300,"xatol":1e-7})
        candidates.append(p.x)
    best=min(candidates,key=lambda p:loss(sim(p,valid),1.2))
    lower,upper=activation_gap((*TRAIN_SCALES,*CHECK_SCALES))
    record=dict(weights=best.tolist(),divisors=DIVISORS.tolist(),boundary=(lower+upper)/2,
                ordinary_max=lower,slalom_min=upper,train=common.stat(sim(best,train)),
                validation=common.stat(sim(best,valid)))
    print(record,flush=True)
    (HERE/"slalom.json").write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":
    main()
