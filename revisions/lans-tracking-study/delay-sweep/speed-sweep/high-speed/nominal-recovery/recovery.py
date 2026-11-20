"""Learn early steering corrections around the nominal reference tracking manifold."""
import ctypes
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_discrete_are
from stable_baselines3 import PPO
import torch

HERE=Path(__file__).resolve().parent
HIGH=HERE.parent
sys.path.insert(0,str(HIGH))
from fit_high import common,loss
from harmonic import make_fit
from nominal_teacher import LocalTeacher

SPEEDS=(120,150)
NODES=8
LABEL_SCALES=((.871,1.039),(1.129,.958),(.963,1.018),(1.044,.974))
TRAIN_SCALES=((.873,1.041),(1.127,.96),(.956,1.025),(1.05,.975))
CHECK_SCALES=((.881,1.042),(1.118,.961),(.96,1.01))
LABEL_SEEDS=range(45000,45014)
TRAIN_SEEDS=range(45500,45514)
CHECK_SEEDS=range(46000,46016)
XS=np.array([.1,.5,.05,.2])


def reference_map(env):
    vx=env.vx
    a,b=env.Ad,env.Bd
    frequencies=np.r_[.15,.35,.7,1.1,1.6,np.pi*vx/25*np.array([.945,.974,1.,1.025,1.051])]
    hs=.01*np.arange(1,5)
    h=frequencies[:,None]*hs
    q=((1+1j*h-np.exp(1j*h))/(10*hs**2))@common.TRANSFORM
    z=np.exp(.01j*frequencies)
    disturbance=(-env.Gd*frequencies[:,None]**2-1j*env.Hd*frequencies[:,None]**3)/vx
    response=np.linalg.solve(z[:,None,None]*np.eye(4)-a,
        np.stack((disturbance,np.broadcast_to(b,disturbance.shape)),axis=-1))
    ff=-response[:,0,0]/response[:,0,1]
    target=response[:,:,0]+response[:,:,1]*ff[:,None]
    weight=1/frequencies**2
    q*=weight[:,None]
    target*=weight[:,None]
    return np.linalg.lstsq(np.vstack((q.real,q.imag)),np.vstack((target.real,target.imag)),rcond=None)[0]


def sampler(env,observations):
    return np.r_[env.x,observations[5:8]]


def collect(speed,start):
    states,commands,weights=[],[],[]
    for i,((amp,scale),path) in enumerate(product(LABEL_SCALES,common.PATHS)):
        env,n,ref=common.configure_env("LANS",path,0,speed,amp,scale,"fixed_road")
        for seed in LABEL_SEEDS:
            for expert in (False,True):
                env.reset(seed=seed+i*83)
                env.ref=ref
                obs=env.obs()
                teacher=LocalTeacher(env)
                for tick in range(16):
                    x=sampler(env,obs)
                    target=teacher.action(env)
                    states.append(x)
                    commands.append(target)
                    weights.append(1. if tick<4 else .7 if tick<9 else .30)
                    u=target if expert else (start@x).clip(-.5,.5)
                    obs,_,failed,_,_=env.step([(u+env.K@env.x)/1.5])
                    if failed:
                        break
    return np.array(states),np.array(commands),np.array(weights)


def simulator():
    lib=ctypes.CDLL(str(HERE/"rollout_recovery.so"))
    sim=lib.recovery
    array=common.FLOAT_ARRAY
    sim.argtypes=[ctypes.c_int]*4+[array]*13+[ctypes.c_void_p]*2
    sim.restype=None
    return sim


def evaluator():
    sim=simulator()
    def replay(policy,data,steps=None):
        n,a,b,fields=data
        rows=np.empty((len(fields[2]),3))
        co,directions,thresholds,values,settled=policy[:5]
        extra=policy[5].ctypes.data if len(policy)>5 and policy[5] is not None else None
        preview=policy[6].ctypes.data if len(policy)>6 and policy[6] is not None else None
        sim(len(rows),n,n if steps is None else steps,len(thresholds),a,b,common.KR,
            fields[0],fields[1],fields[2],common.TRANSFORM,co,directions,thresholds,
            values,settled,rows,extra,preview)
        return rows
    return replay


def as_t(value):
    return torch.as_tensor(value,dtype=torch.float64)


def fit(speed):
    torch.set_num_threads(1)
    source=PPO.load(HIGH/"gn_policy_72.zip",device="cpu")
    training=common.dataset(speed,0,TRAIN_SCALES,TRAIN_SEEDS,source,vary_resets=True)
    validation=common.dataset(speed,0,CHECK_SCALES,CHECK_SEEDS,source,vary_resets=True)
    tail=common.dataset(speed,0,TRAIN_SCALES,range(46320,46323),source,vary_resets=True)
    tail[-1][2][:]=0.
    env,_,_=common.configure_env("LANS","dlc",0,speed,1.,1.,"fixed_road")
    settled=np.ascontiguousarray(reference_map(env))
    evaluate=evaluator()
    analytic=make_fit(env)
    a,b=env.Ad,env.Bd
    empty=np.zeros((NODES,4))
    steady=[]
    for r in (.055,.025,.011,.004,.001):
        P=solve_discrete_are(a,b[:,None],np.diag([1000.,0.,.01,0.]),np.array([[r]]))
        gains=-(b@P@a)/(r+b@P@b)
        steady.append(np.r_[gains,analytic(gains)])
    base=min(steady,key=lambda p:loss(evaluate((p,empty,np.ones(NODES),np.zeros(NODES),settled),tail),.5))
    labels,targets,weight=map(as_t,collect(speed,base))
    stable=as_t(settled)
    scale=as_t(XS)
    K=as_t(common.KR)
    nominal=as_t(base)
    dims=as_t([85.,2.,2.,.17])
    diff=torch.nn.Parameter(torch.zeros(4,dtype=torch.float64))
    recovery=torch.nn.Parameter(as_t([
        [1.,.25,0.,0.],[-1.,-.25,0.,0.],
        [1.,-.7,.04,0.],[-1.,.7,-.04,0.],
        [.35,.9,.04,-.02],[-.35,-.9,-.04,.02],
        [1.,.3,-.08,.035],[-1.,-.3,.08,-.035],
    ]))
    threshold=torch.nn.Parameter(as_t([.065,.065,.12,.12,.033,.033,.22,.22]))
    output=torch.nn.Parameter(as_t([-.02,.02,-.045,.045,-.016,.016,-.04,.04]))
    generator=torch.Generator().manual_seed(speed+614421)
    parameters=[diff,recovery,threshold,output]

    def command(x,q):
        z=(x-q@stable)/scale
        correction=(z@recovery.T-threshold).relu()@output
        linear=x@nominal[:4]+q@nominal[4:]+(z*scale)@(diff*dims)
        offset=-x@K
        return (offset+(linear+correction-offset).clamp(-1.5,1.5)).clamp(-.5,.5)

    def get():
        delta=(diff*dims).detach().numpy()
        adjusted=np.r_[base[:4]+delta,base[4:]-settled@delta]
        return adjusted,recovery.detach().numpy().copy(),threshold.detach().numpy().copy(),output.detach().numpy().copy(),settled

    initial=get()
    snapshots=[initial]
    optimizer=torch.optim.Adam(parameters,lr=.007)
    for iteration in range(1700):
        selected=torch.randint(len(targets),(448,),generator=generator)
        x=labels[selected]
        prediction=command(x[:,:4],x[:,4:])
        value=(weight[selected]*(prediction-targets[selected]).square()).mean()
        value+=1.e-6*diff.square().sum()
        optimizer.zero_grad()
        value.backward()
        optimizer.step()
        with torch.no_grad():
            threshold.clamp_(.008,.9)
            diff.clamp_(-2.,2.)
            recovery.clamp_(-2.5,2.5)
        if iteration+1 in (250,650,1100,1700):
            snapshots.append(get())
    _,a,b,arr=training
    force,refs,x0,_=arr
    nsteps=27
    A,B,F,R,X=map(as_t,(a,b,force[:,:nsteps],refs[:,:nsteps],x0))
    horizon=as_t(.01*np.arange(1,5))
    whiten=as_t(common.TRANSFORM)

    def closed_error(indices):
        x=X[indices]
        reference=R[indices]
        summed=x[:,0].square()
        for t in range(nsteps):
            r=reference[:,t]
            v=r[:,2:3]
            obs=((r[:,:1]+x[:,:1])+v*(r[:,1:2]+x[:,2:3])*horizon-r[:,3:7]).float().double()
            preview=(((obs-x[:,:1]-v*x[:,2:3]*horizon)/(10*horizon.square())).float().double()@whiten).float().double()
            steering=command(x,preview)
            x=x@A.T+steering[:,None]*B+F[indices,t]
            summed=summed+x[:,0].square()
        return (summed+1e-10).sqrt().mean()

    def rank(c):
        return loss(evaluate(c,validation),.8)+.35*loss(evaluate(c,tail),.8)
    chosen=min(snapshots,key=rank)
    with torch.no_grad():
        delta=(chosen[0][:4]-base[:4])/dims.numpy()
        diff.copy_(as_t(delta))
        recovery.copy_(as_t(chosen[1]))
        threshold.copy_(as_t(chosen[2]))
        output.copy_(as_t(chosen[3]))
    optimizer=torch.optim.Adam(parameters,lr=.0016)
    for iteration in range(1500):
        chosen=torch.randint(len(X),(112,),generator=generator)
        value=closed_error(chosen)+.0006*diff.square().sum()
        optimizer.zero_grad()
        value.backward()
        torch.nn.utils.clip_grad_norm_(parameters,.04)
        optimizer.step()
        with torch.no_grad():
            threshold.clamp_(.008,.9)
            diff.clamp_(-2.,2.)
            recovery.clamp_(-2.5,2.5)
            output.clamp_(-6.,6.)
        if iteration+1 in (200,400,700,1000,1250,1500):
            candidate=get()
            snapshots.append(candidate)
            print(speed,iteration+1,round(rank(candidate),9),flush=True)
    best=min(snapshots,key=rank)
    row=dict(speed=speed,weights=best[0].tolist(),directions=best[1].tolist(),thresholds=best[2].tolist(),
             coefficients=best[3].tolist(),reference_map=best[4].tolist(),
             validation=common.stat(evaluate(best,validation)),
             initial_validation=common.stat(evaluate(initial,validation)))
    print(row,flush=True)
    return row


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(fit,SPEEDS))
    (HERE/"recovery_fit.json").write_text(json.dumps(rows,indent=2)+"\n")
