"""Fit four additional recovery hinges in the existing actor for the speed grid."""
import json
from itertools import product

import numpy as np
from stable_baselines3 import PPO
import torch

from fit_high import HERE,common,prepare,loss
from nominal_teacher import LocalTeacher
from precision import features

SPEEDS=(120,150)
TRAIN=((.865,1.039),(1.132,.96),(.934,1.028),(1.052,.971))
CAL=((.887,1.038),(1.121,.96),(.976,1.015))
SCALES=np.array([.1,.5,.05,.2])
LABEL_SEEDS=range(42700,42713)
TRAIN_SEEDS=range(43000,43012)
VAL_SEEDS=range(43100,43113)


def tensor(value):
    return torch.as_tensor(value,dtype=torch.float64)


def collect(speed,incumbent):
    features_out,targets=[],[]
    for (amplitude,ts),path in product(TRAIN[:2],common.PATHS):
        env,_,ref=common.configure_env("LANS",path,0,speed,amplitude,ts,"fixed_road")
        for seed in LABEL_SEEDS:
            for expert in (True,False):
                env.reset(seed=seed+37*common.PATHS.index(path))
                env.ref=ref
                teacher=LocalTeacher(env)
                obs=env.obs()
                for k in range(14):
                    x=features(env,obs)
                    label=teacher.action(env)
                    features_out.append(x)
                    targets.append(label)
                    command=label if expert else float(np.clip(incumbent@x,-.5,.5))
                    obs,_,failed,_,_=env.step([(command+env.K@env.x)/env.residual_scale])
                    if failed:
                        break
    return tensor(np.array(features_out)),tensor(np.asarray(targets))


def calibrated_data():
    model=PPO.load(HERE.parent/"low-speed/tail_policy_72.zip",device="cpu")
    candidates=[json.loads((HERE/f"nominal_{name}.json").read_text()) for name in ("combined","final","gn")]
    accum=[]
    for speed in SPEEDS:
        sim,_,_=prepare(speed,0)
        train=common.dataset(speed,0,TRAIN,TRAIN_SEEDS,model,vary_resets=True)
        check=common.dataset(speed,0,CAL,VAL_SEEDS,model,vary_resets=True)
        choices=[np.array(next(r["weights"] for r in group if r["speed"]==speed)) for group in candidates]
        start=min(choices,key=lambda p:loss(sim(p,train),.7))
        accum.append((sim,train,check,start))
    return accum


def main():
    torch.set_num_threads(1)
    initial=calibrated_data()
    translations=tensor(np.array([item[-1] for item in initial]))
    generator=torch.Generator().manual_seed(852041)
    extra_direction=torch.nn.Parameter(tensor([[1.,.25], [1.,-.40],[.25,1.], [1.1,.74]]))
    cut=torch.nn.Parameter(tensor([.09,.17,.065,.30]))
    old=torch.nn.Parameter(torch.zeros_like(translations))
    added=torch.nn.Parameter(tensor([[.07,-.07,-.04,.04],[.07,-.07,-.04,.04]]))
    scale=tensor([80.,2.,2.,.18,.12,.18,.35,*([1.25]*14)])
    offsets=tensor(SCALES)
    kr=tensor(common.KR)
    directions=tensor(common.DIRECTIONS)
    thresholds=tensor(common.THRESHOLDS)
    hs=tensor(.01*np.arange(1,5))
    decoder=tensor(common.TRANSFORM)
    labels=[collect(speed,item[-1]) for speed,item in zip(SPEEDS,initial,strict=True)]
    q_params=[old,added,extra_direction,cut]

    def coefficients():
        return translations+old*scale

    def add_features(x):
        root=(x[...,:2]/offsets[:2])@extra_direction.T
        return (root-cut).relu()-(-root-cut).relu()

    def apply_limits(raw,x):
        base=-x@kr
        return (base+(raw-base).clamp(-1.5,1.5)).clamp(-.5,.5)

    def checkpoint():
        full=np.column_stack((coefficients().detach().numpy(),added.detach().numpy()))
        d=np.column_stack((extra_direction.detach().numpy(),np.zeros((4,2))))
        h=cut.detach().numpy().copy()
        d=np.ascontiguousarray(np.vstack((common.DIRECTIONS,d)))
        h=np.ascontiguousarray(np.r_[common.THRESHOLDS,h])
        return full,d,h

    initial_weights,d0,h0=checkpoint()
    initial_weights[:,21:]=0.
    saved=[(initial_weights,d0,h0)]
    optimizer=torch.optim.Adam(q_params,lr=.004)
    for step in range(1000):
        objective=torch.zeros((),dtype=torch.float64)
        for i,(x,y) in enumerate(labels):
            index=torch.randint(len(y),(384,),generator=generator)
            obs,label=x[index],y[index]
            raw=obs@coefficients()[i]+add_features(obs)@added[i]
            steering=apply_limits(raw,obs[:,:4])
            objective+=((steering-label)**2).mean()
        objective+=1e-5*old.square().mean()
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            cut.clamp_(.012,.8)
            extra_direction.clamp_(-2.5,2.5)
            old.clamp_(-1.5,1.5)
        if step in (249,499,999):
            saved.append(checkpoint())

    datasets=[]
    for _,train,_,_ in initial:
        n,a,b,arr=train
        force,refs,x0,_=arr
        datasets.append(tuple(tensor(v) for v in (a,b,force[:,:65],refs[:,:65],x0)))
    a,b,disturbance,refs,x0=(torch.stack(values) for values in zip(*datasets,strict=True))

    def rollout_loss(index):
        x=x0[:,index]
        force=disturbance[:,index]
        course=refs[:,index]
        q=coefficients()
        squared=x[...,0].square()
        for step in range(65):
            ref=course[:,:,step]
            vx=ref[...,2:3]
            raw=((x[...,0]+ref[...,0])[...,None]+vx*(x[...,2]+ref[...,1])[...,None]*hs-ref[...,3:])
            preview=(((raw.float().double()-x[...,:1]-vx*x[...,2:3]*hs)/(10*hs.square())).float().double()
                     @decoder).float().double()
            z=(x/offsets).float().double()@directions.T
            basis=torch.cat((x,preview,(z-thresholds).relu()-(-z-thresholds).relu()),dim=-1)
            steering=torch.einsum("sni,si->sn",basis,q)+torch.einsum("sni,si->sn",add_features(x),added)
            u=apply_limits(steering,x)
            x=torch.bmm(x,a.transpose(1,2))+u[...,None]*b[:,None,:]+force[:,:,step]
            squared=squared+x[...,0].square()
        return ((squared+1e-12).sqrt()/np.sqrt(66.)).mean()

    optimizer=torch.optim.Adam(q_params,lr=.0015)
    for step in range(1000):
        index=torch.randint(x0.shape[1],(56,),generator=generator)
        value=rollout_loss(index)
        optimizer.zero_grad()
        value.backward()
        torch.nn.utils.clip_grad_norm_(q_params,.01)
        optimizer.step()
        with torch.no_grad():
            cut.clamp_(.012,.8)
            extra_direction.clamp_(-2.5,2.5)
            old.clamp_(-1.5,1.5)
        if step in (99,249,499,749,999):
            saved.append(checkpoint())
            print("rollout",step+1,float(value.detach()),flush=True)

    def score(item):
        param,d,h=item
        calibration=[]
        for i,(sim,_,check,_) in enumerate(initial):
            result=sim(param[i],check,directions=d,thresholds=h)
            calibration.append(loss(result,.8))
        return sum(calibration)

    p,d,h=min(saved,key=score)
    rows=[]
    for i,(speed,(sim,_,check,incumbent)) in enumerate(zip(SPEEDS,initial,strict=True)):
        row=dict(speed=speed,delay=0,weights=p[i].tolist(),
                 validation=common.stat(sim(p[i],check,directions=d,thresholds=h)),
                 prior_validation=common.stat(sim(incumbent,check)))
        print(row,flush=True)
        rows.append(row)
    previous=json.loads((HERE/"nominal_gn.json").read_text())
    rows.extend(r for r in previous if r["speed"] not in SPEEDS)
    (HERE/"nominal_extra.json").write_text(json.dumps(rows,indent=2)+"\n")
    (HERE/"recovery_features.json").write_text(json.dumps(dict(directions=d[14:].tolist(),
                                                            thresholds=h[14:].tolist()),indent=2)+"\n")


if __name__=="__main__":
    main()
