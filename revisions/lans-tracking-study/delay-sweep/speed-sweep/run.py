"""Replay the frozen long-delay checkpoints on one road course at different speeds."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import lsq_linear
import torch

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parent
sys.path.insert(0, str(SWEEP / "actor-refinement"))
from candidate import CandidateEnv, ORIGINAL_ENV, TRAIN_SEEDS, sweep
from dynamic_env import DT, HORIZON, reference
from matched_mpc import MatchedMPC
from stable_baselines3 import PPO

SPEEDS = (15, 30, 60, 90, 120, 150, 200)
ROAD_DELAYS = (0, 200, 1000)
CONTROLLERS = ("LANS", "PPO", "LQR", "MPC")
SETTINGS = {"fixed_road": ROAD_DELAYS, "fixed_time": (200,)}
POLICIES = {}


class SweepMPC(MatchedMPC):
    def action(self, env):
        try:
            return super().action(env)
        except RuntimeError as error:
            if not str(error).startswith("Matched MPC solve failed:"):
                raise
            # Solve the same quadratic and bounds if OSQP does not converge.
            x = np.zeros(4)
            disturbance = []
            for i in range(self.N):
                tick = min(env.k + i, env.ref.shape[1] - 1)
                x = self.A @ x + (self.G * env.ref[2, tick] + self.H * env.ref[3, tick])/self.vx
                disturbance.extend(x)
            c = self.SW @ (self.T @ env.x + disturbance)
            c[0] -= .01 * env.prev_u
            factor = cholesky(self.Hessian, lower=True)
            target = -solve_triangular(factor, c, lower=True)
            result = lsq_linear(factor.T, target, bounds=(-.5, .5), method="bvls", tol=1e-10)
            if not result.success:
                raise RuntimeError(result.message) from error
            self.last = np.r_[result.x[1:], result.x[-1]]
            return float(result.x[0])


def configure_env(controller, path, delay, speed, amplitude, scale, profile):
    vx = speed/3.6
    env_type = CandidateEnv if controller == "LANS" else ORIGINAL_ENV
    env = env_type(kind=controller.lower(), path=path, delay_mode=f"fixed{delay}",
                   vx=vx, mass=1650., amplitude=amplitude, time_scale=scale)
    factor = vx/25 if profile == "fixed_road" else 1.
    horizon = round(HORIZON/factor)
    reference_scale = scale*factor
    ref = np.asarray(reference(path, DT*np.arange(horizon + 6)*reference_scale))
    ref *= amplitude * reference_scale**np.arange(4)[:, None]
    return env, horizon, ref


def run(task):
    profile, speed, delay, controller, seed, path = task
    model = POLICIES.get((controller, seed))
    results = []
    for amplitude, scale in sweep.P["test"]["scales"]:
        env, horizon, ref = configure_env(controller, path, delay, speed, amplitude, scale, profile)
        peak_ref_acceleration = float(np.max(np.abs(ref[2, :horizon])))
        for physical_seed in sweep.P["test"]["seeds"]:
            env.reset(seed=physical_seed)
            env.ref = ref
            env.delays = np.full(horizon + 1, delay, dtype=float)
            obs = env.obs()
            error = float(env.x[0])**2
            peak = abs(float(env.x[0]))
            peak_heading = abs(float(env.x[2]))
            saturations = 0
            applied_saturations = 0
            mpc = SweepMPC(mass=env.mass, vx=env.vx) if controller == "MPC" else None
            for step in range(1, horizon + 1):
                if model is not None:
                    action = model.predict(obs, deterministic=True)[0]
                elif mpc is not None:
                    action = [mpc.action(env)/.5]
                else:
                    action = [0.]
                # A fixed-road run ends at its speed-dependent distance horizon.
                obs, _, failed, _, info = env.step(action)
                ey, epsi = info["error"], info["heading_error"]
                error += ey**2
                peak = max(peak, abs(ey))
                peak_heading = max(peak_heading, abs(epsi))
                saturations += int(info["saturation"])
                applied_saturations += int(abs(info["applied"]) >= .5)
                if failed:
                    break
            completed = not failed
            assert isfinite(error)
            penalty = (horizon - step) * max(2.5, abs(ey))**2
            results.append(dict(
                profile=profile, speed_kmh=speed, delay_ms=delay, controller=controller,
                policy_seed=seed, path=path, physical_seed=physical_seed,
                amplitude=amplitude, time_scale_at_90=scale, planned_steps=horizon, steps=step,
                completed=int(completed), rmse_penalized_m=sqrt((error + penalty)/(horizon + 1)),
                rmse_observed_m=sqrt(error/(step + 1)), peak_lateral_m=peak,
                peak_heading_rad=peak_heading, saturation_fraction=saturations/step,
                applied_saturation_fraction=applied_saturations/step,
                reference_peak_lateral_acceleration_mps2=peak_ref_acceleration,
                failed_on_lateral=int(failed and abs(ey)>2.5),
                failed_on_heading=int(failed and abs(epsi)>.5),
                mpc_fallbacks=mpc.failures if mpc is not None else 0,
            ))
    return results


def save(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(episodes, by_path):
    groups = {}
    for row in episodes:
        key = (row["profile"], row["speed_kmh"], row["delay_ms"])
        if by_path:
            key += (row["path"],)
        groups.setdefault(key, {}).setdefault(row["controller"], []).append(row)
    rows = []
    for key, controllers in sorted(groups.items()):
        group = dict(zip(("profile", "speed_kmh", "delay_ms", "path"), key))
        scores = {c: mean(r["rmse_penalized_m"] for r in runs) for c, runs in controllers.items()}
        for controller in sorted(controllers, key=scores.get):
            runs = controllers[controller]
            rmse = scores[controller]
            rows.append(dict(
                **group, controller=controller, rank=1+sum(v<rmse-1e-12 for v in scores.values()),
                rmse_penalized_m=rmse, rmse_observed_m=mean(r["rmse_observed_m"] for r in runs),
                completed=sum(r["completed"] for r in runs), episodes=len(runs),
                mean_saturation_fraction=mean(r["saturation_fraction"] for r in runs),
                mean_applied_saturation_fraction=mean(r["applied_saturation_fraction"] for r in runs),
                mpc_fallbacks=sum(r["mpc_fallbacks"] for r in runs),
            ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-high-speed", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for seed in TRAIN_SEEDS:
        POLICIES["LANS", seed] = PPO.load(SWEEP / "completion-followup" / f"final_policy_{seed}.zip", device="cpu")
        POLICIES["PPO", seed] = sweep.load("limited_ppo", seed)
    jobs = []
    settings = {"fixed_road": (200,)} if args.fresh_high_speed else SETTINGS
    if args.fresh_high_speed:
        sweep.P["test"] = {**sweep.P["test"], "seeds": list(range(33800, 33805))}
    for profile, delays in settings.items():
        # The identical 90 km/h steady-course case is already in fixed_road.
        speeds = [v for v in SPEEDS if profile == "fixed_road" or v != 90]
        if args.fresh_high_speed:
            speeds = [120, 150, 200]
        for speed, delay, controller, path in product(speeds, delays, CONTROLLERS, sweep.P["test"]["families"]):
            for seed in TRAIN_SEEDS if controller in ("LANS", "PPO") else (-1,):
                jobs.append((profile, speed, delay, controller, seed, path))
    episodes = []
    progress = {}
    with ProcessPoolExecutor(max_workers=2 if args.fresh_high_speed else 4) as pool:
        task_for = {pool.submit(run, job): job for job in jobs}
        for pending in as_completed(task_for):
            job = task_for[pending]
            batch = pending.result()
            episodes.extend(batch)
            key = job[:3]
            progress[key] = progress.get(key, 0) + len(batch)
            if progress[key] == 320:
                print(f"{key[0]} {key[1]} km/h {key[2]} ms: 320/320 evaluated", flush=True)
    if args.fresh_high_speed:
        assert len(episodes) == 960
    else:
        assert len(episodes) == 320 * (len(SPEEDS)*len(ROAD_DELAYS) + len(SPEEDS)-1)
        episodes.extend({**r, "profile": "fixed_time"} for r in tuple(episodes)
                        if r["profile"]=="fixed_road" and r["speed_kmh"]==90 and r["delay_ms"]==200)
    episodes.sort(key=lambda row: tuple(row[name] for name in
                  ("profile", "speed_kmh", "delay_ms", "controller", "policy_seed",
                   "path", "amplitude", "time_scale_at_90", "physical_seed")))
    prefix = "fresh_" if args.fresh_high_speed else ""
    save(f"{prefix}episodes.csv", episodes)
    save(f"{prefix}rank_by_speed_and_delay.csv", summarize(episodes, by_path=False))
    save(f"{prefix}rank_by_path.csv", summarize(episodes, by_path=True))


if __name__ == "__main__":
    main()
