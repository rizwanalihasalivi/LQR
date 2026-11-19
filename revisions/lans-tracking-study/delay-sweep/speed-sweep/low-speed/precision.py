"""Fit bounded offline steering targets for the 15 km/h nominal condition."""
import json

import numpy as np
from scipy.optimize import lsq_linear, least_squares, minimize
from stable_baselines3 import PPO

from fit import (HERE, SWEEP, PATHS, DIRECTIONS, THRESHOLDS, configure_env, dataset, prepare,
                 stat, torch)


def features(env, obs):
    v = DIRECTIONS @ obs[:4]
    return np.r_[env.x, obs[5:8],
                 np.maximum(v - THRESHOLDS,0.) - np.maximum(-v - THRESHOLDS,0.)]


class Teacher:
    def __init__(self, env):
        horizon = 28
        a, b = env.Ad, env.Bd
        self.a = np.stack([np.linalg.matrix_power(a, j+1) for j in range(horizon)])
        self.matrix = np.zeros((2*horizon, horizon))
        for k in range(horizon):
            for j in range(k+1):
                response = np.linalg.matrix_power(a,k-j)@b
                self.matrix[k,j] = response[0]
                self.matrix[k+horizon,j] = .002*response[2]
        self.matrix = np.vstack((self.matrix, np.eye(horizon)*.000025))
        self.disturbance = (env.ref[2,:40,None]*env.Gd+env.ref[3,:40,None]*env.Hd)/env.vx
        self.a_plant = a

    def command(self, x, tick):
        steps = len(self.a)
        free = self.a@x
        extra = np.zeros(4)
        for j in range(steps):
            extra = self.a_plant@extra + self.disturbance[min(len(self.disturbance)-1,tick+j)]
            free[j] += extra
        target = -np.r_[free[:,0], .002*free[:,2], np.zeros(steps)]
        solution = lsq_linear(self.matrix, target, bounds=(-.5,.5), method="bvls",
                               max_iter=100, tol=1e-10)
        return float(solution.x[0])


def collect(incumbent):
    basis, targets, importance = [], [], []
    for scale_index,(amplitude,scale) in enumerate(((.86,1.041),(1.13,.959))):
        for path_index,path in enumerate(PATHS):
            env,_,ref=configure_env("LANS",path,0,15,amplitude,scale,"fixed_road")
            for seed in range(37230,37238):
                seed += 80*scale_index+12*path_index
                for teacher_path in (True,False):
                    env.reset(seed=seed)
                    env.ref=ref
                    obs=env.obs()
                    teacher=Teacher(env)
                    for tick in range(16):
                        weights=features(env,obs)
                        label=teacher.command(env.x,tick)
                        basis.append(weights)
                        targets.append(label)
                        importance.append(1. if tick<5 else .4)
                        command=label if teacher_path else float(np.clip(incumbent@weights,-.5,.5))
                        action=(command+env.K@env.x)/1.5
                        obs, _, terminated, _, _=env.step([action])
                        assert not terminated
    return np.array(basis),np.array(targets),np.sqrt(np.array(importance))


def main():
    torch.set_num_threads(1)
    record = next(r for r in json.loads((HERE/"nominal_polish.json").read_text()) if r["speed"]==15)
    original=np.array(record["weights"])
    basis,desired,weight=collect(original)
    evaluate,_,_=prepare(15,0)
    model=PPO.load(SWEEP/"completion-followup/final_policy_72.zip",device="cpu")
    train=dataset(15,0,((.878,1.039),(1.106,.961),(1.02,.99)),range(37650,37662),model,
                  limit=110,vary_resets=True)
    valid=dataset(15,0,((.864,1.046),(1.115,.955)),range(39130,39138),model,vary_resets=True)
    tail=dataset(15,0,((.89,1.04),(1.105,.96)),[39600],model)
    tail[-1][2][:]=0.
    width=np.r_[30.,1.45,2.5,.18,.5,.4,.4,np.full(14,1.8)]
    bounds=(original-width,original+width)
    trials=[original]
    for regularizer in (.002,.008):
        def residual(p):
            steering=(basis@p).clip(-.5,.5)
            return np.r_[weight*(steering-desired),regularizer*(p-original)/width]
        trained=least_squares(residual,original,bounds=bounds,max_nfev=160,x_scale=width)
        trials.append(trained.x)
    def early_error(p):
        return evaluate(p,train)[:,0].mean()
    choices=sorted(trials,key=early_error)[:2]
    for start in choices:
        result=minimize(early_error,start,method="Nelder-Mead",bounds=list(zip(*bounds)),
                        options={"maxiter":1800,"xatol":1e-7})
        trials.append(result.x)
    for p in trials.copy():
        def tail_error(coeff):
            candidate=p.copy()
            candidate[4:7]=coeff
            return evaluate(candidate,tail)[:,0].mean()
        tuned=minimize(tail_error,p[4:7],method="Nelder-Mead",bounds=list(zip(*bounds))[4:7],
                       options={"maxiter":180,"xatol":1e-7})
        candidate=p.copy()
        candidate[4:7]=tuned.x
        trials.append(candidate)
    selected=min(trials,key=lambda p:evaluate(p,valid)[:,0].mean())
    result=dict(speed=15,delay=0,weights=selected.tolist(),validation=stat(evaluate(selected,valid)),
                baseline=stat(evaluate(original,valid)),offline_samples=len(desired))
    print(result,flush=True)
    other=[r for r in json.loads((HERE/"nominal_polish.json").read_text()) if r["speed"]!=15]
    (HERE/"nominal_precision.json").write_text(json.dumps([result,*other],indent=2)+"\n")


if __name__=="__main__":
    main()
