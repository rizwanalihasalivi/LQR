"""Run the unchanged scalar plant with batched deterministic policy inference."""
from math import isfinite, sqrt

import numpy as np

import run as speed_sweep


def run(job):
    profile, speed, delay, controller, seed, path = job
    model = speed_sweep.POLICIES.get((controller,seed))
    if model is None:
        return speed_sweep.run(job)
    results=[]
    for amplitude, scale in speed_sweep.sweep.P["test"]["scales"]:
        initial_seeds=speed_sweep.sweep.P["test"]["seeds"]
        envs=[]
        observations=[]
        for physical_seed in initial_seeds:
            env,horizon,ref=speed_sweep.configure_env(controller,path,delay,speed,amplitude,scale,profile)
            env.reset(seed=physical_seed)
            env.ref=ref
            env.delays=np.full(horizon+1,delay,dtype=float)
            observations.append(env.obs())
            envs.append(env)
        obs=np.array(observations)
        errors=[float(env.x[0])**2 for env in envs]
        peaks=[abs(float(env.x[0])) for env in envs]
        headings=[abs(float(env.x[2])) for env in envs]
        saturation=[0 for _ in envs]
        applied=[0 for _ in envs]
        active=list(range(len(envs)))
        peak_acceleration=float(np.max(np.abs(ref[2,:horizon])))
        for step in range(1,horizon+1):
            actions,_=model.predict(obs[active],deterministic=True)
            remaining=[]
            for i, action in zip(active,actions,strict=True):
                obs[i],_,failed,_,info=envs[i].step(action)
                ey,epsi=info["error"],info["heading_error"]
                errors[i]+=ey**2
                peaks[i]=max(peaks[i],abs(ey))
                headings[i]=max(headings[i],abs(epsi))
                saturation[i]+=int(info["saturation"])
                applied[i]+=int(abs(info["applied"])>=.5)
                if not failed and step<horizon:
                    remaining.append(i)
                    continue
                assert isfinite(errors[i])
                penalty=(horizon-step)*max(2.5,abs(ey))**2
                results.append(dict(
                    profile=profile,speed_kmh=speed,delay_ms=delay,controller=controller,
                    policy_seed=seed,path=path,physical_seed=initial_seeds[i],
                    amplitude=amplitude,time_scale_at_90=scale,planned_steps=horizon,steps=step,
                    completed=int(not failed),rmse_penalized_m=sqrt((errors[i]+penalty)/(horizon+1)),
                    rmse_observed_m=sqrt(errors[i]/(step+1)),peak_lateral_m=peaks[i],
                    peak_heading_rad=headings[i],saturation_fraction=saturation[i]/step,
                    applied_saturation_fraction=applied[i]/step,
                    reference_peak_lateral_acceleration_mps2=peak_acceleration,
                    failed_on_lateral=int(failed and abs(ey)>2.5),
                    failed_on_heading=int(failed and abs(epsi)>.5),mpc_fallbacks=0,
                ))
            active=remaining
            if not active:
                break
    return results
