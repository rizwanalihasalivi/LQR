"""Fit offline bounded steering plans and recovery commands using the existing actor features."""
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import least_squares,lsq_linear,minimize
from stable_baselines3 import PPO

from fit_high import HERE,SPEEDS,prepare,common,loss
from precision import features

SCALES=((.873,1.043),(1.12,.958),(.98,.997))
SEEDS=range(41430,41434)
CHECK_SEEDS=range(41510,41517)
CHECK_SCALES=((.882,1.039),(1.126,.963),(1.01,1.01))


class FullTrajectory:
    def __init__(self,env,horizon):
        self.horizon=horizon
        self.a=env.Ad
        self.b=env.Bd
        h=horizon
        dynamics=sparse.lil_matrix((4*h,5*h))
        for t in range(h):
            row=4*t
            dynamics[row:row+4,row:row+4]=np.eye(4)
            dynamics[row:row+4,4*h+t]=-self.b[:,None]
            if t:
                dynamics[row:row+4,row-4:row]=-self.a
        constraint=sparse.vstack((dynamics.tocsc(),sparse.eye(5*h)),format="csc")
        difference=sparse.eye(h,format="csc")-sparse.eye(h,k=-1,format="csc")
        cost=sparse.block_diag((sparse.diags(np.tile([1.,0.,.000015,0.],h)),
                                .000002*sparse.eye(h)+.000002*(difference.T@difference)),format="csc")
        lower=np.r_[np.tile([-2.5,-np.inf,-.5,-np.inf],h),np.full(h,-.5)]
        upper=-lower
        self.lower=lower
        self.upper=upper
        self.solver=osqp.OSQP()
        self.solver.setup(P=cost*2,q=np.zeros(5*h),A=constraint,
                          l=np.r_[np.zeros(4*h),lower],u=np.r_[np.zeros(4*h),upper],
                          eps_abs=1e-7,eps_rel=1e-7,max_iter=30000,polishing=True,verbose=False)

    def solve(self,env):
        h=self.horizon
        disturbance=(env.ref[2,:h,None]*env.Gd+env.ref[3,:h,None]*env.Hd)/env.vx
        disturbance[0]+=env.Ad@env.x
        d=disturbance.ravel()
        self.solver.update(l=np.r_[d,self.lower],u=np.r_[d,self.upper])
        result=self.solver.solve(raise_error=False)
        if result.info.status_val not in (1,2):
            return None,None
        states=np.vstack((env.x,result.x[:4*h].reshape(-1,4)))
        commands=np.clip(result.x[-h:],-.5,.5)
        if np.max(abs(states[1:,0]))>2.50001 or np.max(abs(states[1:,2]))>.50001:
            return None,None
        return states,commands


class LocalTeacher:
    def __init__(self,env):
        self.a=env.Ad
        z=32
        response=np.zeros((z,4,z))
        self.transition=np.stack([np.linalg.matrix_power(self.a,t+1) for t in range(z)])
        for t in range(z):
            for j in range(t+1):
                response[t,:,j]=np.linalg.matrix_power(self.a,t-j)@env.Bd
        self.matrix=np.vstack((response[:,0],.003*response[:,2],np.eye(z)*.0007))
        self.factor=cho_factor(self.matrix.T@self.matrix)
        self.force=(env.ref[2,:,None]*env.Gd+env.ref[3,:,None]*env.Hd)/env.vx

    def action(self,env):
        future=self.transition@env.x
        state=np.zeros(4)
        n=len(future)
        for t in range(n):
            state=self.a@state+self.force[min(env.k+t,len(self.force)-1)]
            future[t]+=state
        target=-np.r_[future[:,0],.003*future[:,2],np.zeros(n)]
        guess=cho_solve(self.factor,self.matrix.T@target)
        if max(abs(guess))<=.5:
            return float(guess[0])
        fit=lsq_linear(self.matrix,target,bounds=(-.5,.5),method="bvls",tol=1e-9,max_iter=80)
        return float(fit.x[0])


def collect(speed,reference):
    data,tgt,weights=[],[],[]
    full=None
    planned=0
    for (amplitude,time_scale),path in product(SCALES,common.PATHS):
        env,horizon,ref=common.configure_env("LANS",path,0,speed,amplitude,time_scale,"fixed_road")
        if full is None:
            full=FullTrajectory(env,horizon)
        for seed in SEEDS:
            env.reset(seed=seed)
            env.ref=ref
            states,commands=full.solve(env)
            if states is not None:
                planned+=1
                indexes=[*range(10),*range(10,horizon,5)]
                for i in indexes:
                    env.k=i
                    env.x=states[i]
                    data.append(features(env,env.obs()))
                    tgt.append(commands[i])
                    weights.append(2. if i<10 else 1.5 if path=="slalom" else 1.)
            env.reset(seed=seed)
            env.ref=ref
            obs=env.obs()
            local=LocalTeacher(env)
            for t in range(horizon):
                f=features(env,obs)
                if t<8 or t%24==0:
                    data.append(f)
                    tgt.append(local.action(env))
                    weights.append(.7 if t<8 else .2)
                command=float(np.clip(reference@f,-.5,.5))
                action=(command+env.K@env.x)/env.residual_scale
                obs,_,failed,_,_=env.step([action])
                if failed:
                    break
    return np.asarray(data),np.asarray(tgt),np.sqrt(weights),planned


def fit(speed):
    common.torch.set_num_threads(1)
    old=json.loads((HERE/"nominal.json").read_text())
    start=np.array(next(r["weights"] for r in old if r["speed"]==speed))
    feature,target,weight,planned=collect(speed,start)
    evaluate,train,valid=prepare(speed,0)
    model=PPO.load(HERE.parent/"low-speed/tail_policy_72.zip",device="cpu")
    confirm=common.dataset(speed,0,CHECK_SCALES,CHECK_SEEDS,model,vary_resets=True)
    candidates=[start]
    width=np.r_[160.,5.,7.,.8,1.8,3.,3.,np.full(14,6.)]
    bounds=(start-width,start+width)
    for regularizer in (.0008,.006,.016):
        def residual(p):
            command=feature@p
            return np.r_[weight*(command.clip(-.5,.5)-target),
                         .015*weight*(command-target),regularizer*(p-start)/width]
        result=least_squares(residual,start,bounds=bounds,max_nfev=150,x_scale=width)
        candidates.append(result.x)
    def objective(p):
        return loss(evaluate(p,train),.85)
    for init in sorted(candidates,key=objective)[:3]:
        result=minimize(objective,init,method="Nelder-Mead",bounds=list(zip(*bounds)),
                        options={"maxiter":1100,"xatol":1e-7})
        candidates.append(result.x)
    for base in sorted(candidates,key=objective)[:3]:
        def ff(tail):
            p=base.copy()
            p[4:7]=tail
            return objective(p)
        result=minimize(ff,base[4:7],bounds=list(zip(*bounds))[4:7],method="Nelder-Mead",
                        options={"maxiter":150,"xatol":1e-6})
        p=base.copy()
        p[4:7]=result.x
        candidates.append(p)
    best=min(candidates,key=lambda p:(loss(evaluate(p,valid),.85)+loss(evaluate(p,confirm),.85))/2)
    record=dict(speed=speed,delay=0,weights=best.tolist(),labels=len(target),solved_plans=planned,
                validation=common.stat(evaluate(best,valid)),previous_validation=common.stat(evaluate(start,valid)),
                secondary_validation=common.stat(evaluate(best,confirm)))
    print(record,flush=True)
    return record


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(fit,SPEEDS))
    (HERE/"nominal_teacher.json").write_text(json.dumps(rows,indent=2)+"\n")
