"""Check the theoretical nominal lower bound with and without the retained first command."""
import csv
from statistics import mean

import numpy as np
from scipy.optimize import lsq_linear
from stable_baselines3 import PPO
import torch

from fit import HERE, SWEEP, PATHS, SPEEDS
from run import configure_env, sweep, save
import run as speed_sweep


def bound(matrix, disturbance, initial, horizon, first=None):
    if first is not None:
        disturbance = disturbance + matrix[:, 0]*first
        matrix = matrix[:, 1:]
    result = lsq_linear(matrix, -disturbance, bounds=(-.5,.5), method="bvls",
                        max_iter=400, tol=1e-10)
    error = matrix@result.x + disturbance
    slope = 2*matrix.T@error
    correction = slope @ (-.5*np.sign(slope) - result.x)
    return np.sqrt((initial**2 + max(0., error@error+correction))/(horizon+1))


def main():
    torch.set_num_threads(1)
    horizon=80
    predictor = PPO.load(SWEEP / "completion-followup/final_policy_72.zip",device="cpu")
    original=list(csv.DictReader((HERE.parent/"rank_by_speed_and_delay.csv").open()))
    report=[]
    for speed in SPEEDS:
        bounds, retained = [], []
        for amp, scale in sweep.P["test"]["scales"]:
            for path in PATHS:
                env, length, ref = configure_env("LANS",path,0,speed,amp,scale,"fixed_road")
                impulse=[(np.linalg.matrix_power(env.Ad,k)@env.Bd)[0] for k in range(horizon)]
                idx=np.arange(horizon)
                lags=idx[:,None]-idx[None,:]
                response=np.where(lags>=0,np.asarray(impulse)[lags],0.)
                for seed in sweep.P["test"]["seeds"]:
                    env.reset(seed=seed)
                    env.ref=ref
                    state=env.x.copy()
                    first=env.compose(predictor.predict(env.obs(),deterministic=True)[0],0)[0]
                    trace=[]
                    for tick in range(horizon):
                        dd,ddd=ref[2:4,tick]
                        state=env.Ad@state+(env.Gd*dd+env.Hd*ddd)/env.vx
                        trace.append(state[0])
                    bounds.append(bound(response,np.asarray(trace),env.x[0],length))
                    retained.append(bound(response,np.asarray(trace),env.x[0],length,first))
        mpc=next(float(r["rmse_penalized_m"]) for r in original if
                 (r["profile"],int(r["speed_kmh"]),r["delay_ms"],r["controller"])==
                 ("fixed_road",speed,"0","MPC"))
        report.append(dict(speed_kmh=speed,physical_episodes=len(bounds),prefix_steps=horizon,
                           mpc_rmse_m=mpc,free_lower_bound_m=mean(bounds),
                           retained_first_action_lower_bound_m=mean(retained)))
    speed_sweep.HERE=HERE
    save("nominal_lower_bounds.csv",report)


if __name__=="__main__":
    main()
