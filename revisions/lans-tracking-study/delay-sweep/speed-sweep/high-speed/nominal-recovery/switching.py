"""Learn the actuator switching region with a nested ReLU steering correction."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json

import numpy as np
from scipy.optimize import minimize
from stable_baselines3 import PPO
import torch

from recovery import HERE,HIGH,SPEEDS,XS,common,evaluator,loss,as_t,sampler
from nominal_teacher import LocalTeacher

GEOMETRY=((.88,1.04),(1.12,.96))
NEARBY=((.879,1.042),(1.119,.961))
LABEL_SEEDS=range(50510,50528)
ROLLOUT_SEEDS=range(51710,51754)
CALIBRATION_SEEDS=range(52800,52840)
INDEPENDENT_SEEDS=range(53500,53537)
BREAKS=np.array([.10,.28,.56,1.0])


def previous(speed):
    rows=json.loads((HERE/"polished_fit.json").read_text())
    row=next(r for r in rows if r["speed"]==speed)
    keys=("weights","directions","thresholds","coefficients","reference_map")
    return tuple(np.ascontiguousarray(row[k],dtype=float) for k in keys)


def labels(speed,source):
    data,targets,importance=[],[],[]
    p,w,th,a,reference_map=source
    for slot,((amplitude,scale),path) in enumerate(product(GEOMETRY,common.PATHS)):
        env,_,ref=common.configure_env("LANS",path,0,speed,amplitude,scale,"fixed_road")
        for seed in LABEL_SEEDS:
            for mode in ("expert","learner"):
                env.reset(seed=seed+73*slot)
                env.ref=ref
                obs=env.obs()
                teacher=LocalTeacher(env)
                for k in range(13):
                    measurement=sampler(env,obs)
                    target=teacher.action(env)
                    x,q=measurement[:4],measurement[4:]
                    old=p@measurement+np.maximum(w@((x-q@reference_map)/XS)-th,0.)@a
                    action=target if mode=="expert" else float(np.clip(old,-.5,.5))
                    data.append(measurement)
                    targets.append(target)
                    importance.append(1. if k<5 else .36)
                    obs,_,fail,_,_=env.step([(action+env.K@env.x)/env.residual_scale])
                    if fail:
                        break
    return tuple(as_t(np.array(v)) for v in (data,targets,importance))


def fit(speed):
    torch.set_num_threads(1)
    source=previous(speed)
    p0,w0,t0,c0,map0=source
    model=PPO.load(HIGH/"gn_policy_72.zip",device="cpu")
    sim=evaluator()
    data=common.dataset(speed,0,GEOMETRY,ROLLOUT_SEEDS,model,vary_resets=True)
    calibration=common.dataset(speed,0,GEOMETRY,CALIBRATION_SEEDS,model,vary_resets=True)
    independent=common.dataset(speed,0,NEARBY,INDEPENDENT_SEEDS,model,vary_resets=True)
    observation,command,weights=labels(speed,source)
    root=as_t(p0)
    steady=as_t(map0)
    normalization=as_t(XS)
    kr=as_t(common.KR)
    breaks=as_t(BREAKS)
    gain_scale=as_t([46.,1.6,1.2,.10])
    gain=torch.nn.Parameter(torch.zeros(4,dtype=torch.float64))
    direction=torch.nn.Parameter(as_t(w0).clone())
    threshold=torch.nn.Parameter(as_t(t0).clone())
    response=torch.nn.Parameter(as_t(c0).clone())
    extras=torch.nn.Parameter(torch.zeros((8,8),dtype=torch.float64))
    parameters=[gain,direction,threshold,response,extras]
    g=torch.Generator().manual_seed(902813+speed)

    def steer(x,q):
        centered=x-q@steady
        z=(x[...,0]+.008*x[...,1])/.01
        f=torch.stack(((z[...,None]-breaks).relu(),(-z[...,None]-breaks).relu()),dim=-1).flatten(-2)
        raw=x@root[:4]+q@root[4:]+centered@(gain*gain_scale)
        raw+=((centered/normalization)@direction.T+f@extras.T-threshold).relu()@response
        b=-x@kr
        return (b+(raw-b).clamp(-1.5,1.5)).clamp(-.5,.5)

    def checkpoint():
        delta=(gain*gain_scale).detach().numpy()
        p=np.r_[p0[:4]+delta,p0[4:]-map0@delta]
        return (p,direction.detach().numpy().copy(),threshold.detach().numpy().copy(),
                response.detach().numpy().copy(),map0,extras.detach().numpy().copy())

    def constraints():
        with torch.no_grad():
            gain.clamp_(-2.,2.)
            threshold.clamp_(.004,1.5)
            direction.clamp_(-3.,3.)
            response.clamp_(-3.,3.)
            extras.clamp_(-3.,3.)

    def restore(policy):
        with torch.no_grad():
            p,w,th,c,_,u=policy
            gain.copy_(as_t((p[:4]-p0[:4])/gain_scale.numpy()))
            direction.copy_(as_t(w))
            threshold.copy_(as_t(th))
            response.copy_(as_t(c))
            extras.copy_(as_t(u))

    def check(policy):
        return sum(loss(sim(policy,group),.8) for group in (calibration,independent))

    candidates=[checkpoint()]
    optimizer=torch.optim.Adam(parameters,lr=.0022)
    for step in range(1400):
        i=torch.randint(len(command),(640,),generator=g)
        x=observation[i]
        fitted=steer(x[:,:4],x[:,4:])
        objective=(weights[i]*(fitted-command[i]).square()).mean()+.000001*gain.square().mean()
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        constraints()
        if step+1 in (280,560,840,1120,1400):
            candidates.append(checkpoint())

    width=28
    _,a,b,fields=data
    f,r,x,_=fields
    A,B,F,R,X=map(as_t,(a,b,f[:,:width],r[:,:width],x))
    times=as_t(.01*np.arange(1,5))
    transform=as_t(common.TRANSFORM)
    restore(min(candidates,key=check))
    optimizer=torch.optim.Adam(parameters,lr=.0009)
    for step in range(2600):
        selected=torch.randint(len(X),(112,),generator=g)
        current=X[selected]
        references=R[selected]
        applied=F[selected]
        error=current[:,0].square()
        for tick in range(width):
            pose=references[:,tick]
            vx=pose[:,2:3]
            project=((pose[:,:1]+current[:,:1])+vx*(pose[:,1:2]+current[:,2:3])*times-pose[:,3:7])
            cleaned=(project.float().double()-current[:,:1]-vx*current[:,2:3]*times)/(10*times.square())
            q=(cleaned.float().double()@transform).float().double()
            u=steer(current,q)
            current=current@A.T+B*u[:,None]+applied[:,tick]
            error=error+current[:,0].square()
        objective=(error+1e-12).sqrt().mean()+.0001*gain.square().mean()
        optimizer.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(parameters,.06)
        optimizer.step()
        constraints()
        if step+1 in (400,800,1200,1600,2000,2600):
            candidate=checkpoint()
            candidates.append(candidate)
            print(speed,step+1,check(candidate),flush=True)

    parent=min(candidates,key=check)
    p,d,t,c,m,u=parent
    scales=np.r_[34.,.45,.42,.06,np.full(len(c),.33)]
    def tuned(z):
        delta=z[:4]*scales[:4]
        new=np.r_[p[:4]+delta,p[4:]-m@delta]
        return new,d,t,c+z[4:]*scales[4:],m,u
    for n in (10,22,40):
        def score(z):
            return loss(sim(tuned(z),data,steps=n),.8)
        result=minimize(score,np.zeros(len(scales)),method="Powell",bounds=[(-1.,1.)]*len(scales),
                        options={"maxfev":2700,"ftol":1e-10,"xtol":1e-6})
        candidates.append(tuned(result.x))
    winner=min(candidates,key=check)
    names=("weights","directions","thresholds","coefficients","reference_map","switching")
    row=dict(speed=speed,**{k:v.tolist() for k,v in zip(names,winner,strict=True)},
             validation=common.stat(sim(winner,calibration)),
             initial_validation=common.stat(sim(source,calibration)),
             secondary_validation=common.stat(sim(winner,independent)))
    print(speed,row["validation"],row["initial_validation"],row["secondary_validation"],flush=True)
    return row


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        result=list(pool.map(fit,SPEEDS))
    (HERE/"switching_fit.json").write_text(json.dumps(result,indent=2)+"\n")
