"""Evaluate the frozen road-test controllers under seeded measurement noise."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from itertools import product
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from stable_baselines3 import PPO
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from benchmark import RECOVERY, FRESH_SEEDS, configure, run as batched_run, sweep

ROADS = ("straight", "step", "dlc", "curve")
SPEEDS = (90, 150)
DELAYS = (0, 200, 1000)
# Units are m, m/s, rad, rad/s in the unscaled state.
LEVELS = {"mild": (.01, .02, .002, .01), "moderate": (.05, .10, .01, .05)}


def write(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_policies():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for seed in sweep.TRAIN_SEEDS:
        sweep.POLICIES["LANS", seed] = PPO.load(RECOVERY / f"switching_policy_{seed}.zip", device="cpu")
        sweep.POLICIES["PPO", seed] = sweep.sweep.load("limited_ppo", seed)


def measured_input(env, horizon, sigma):
    reset, observe, compose = env.reset, env.obs, env.compose
    noise = np.zeros((horizon + 1, 4))

    def measured(call, *args):
        true_state = env.x
        env.x = true_state + noise[env.k]
        try:
            return call(*args)
        finally:
            env.x = true_state

    def reset_with_noise(*, seed, options=None):
        rng = np.random.default_rng(np.random.SeedSequence([61041, seed]))
        noise[:] = rng.normal(size=noise.shape) * sigma
        return reset(seed=seed, options=options)

    env.reset = reset_with_noise
    env.obs = lambda: measured(observe)
    env.compose = lambda action, delay: measured(compose, action, delay)
    env.measured_call = measured


class MeasuredMPC(sweep.SweepMPC):
    def action(self, env):
        return env.measured_call(super().action, env)


def replay(job):
    level, speed, delay, path, controller, seed = job

    def noisy_configure(*args):
        env, horizon, ref = configure(*args)
        measured_input(env, horizon, LEVELS[level])
        return env, horizon, ref

    sweep.configure_env = noisy_configure
    sweep.SweepMPC = MeasuredMPC
    return [dict(noise=level, **row) for row in batched_run(
        ("fixed_road", speed, delay, controller, seed, path))]


def summarize(episodes):
    levels = ("clean", *LEVELS)
    cells = []
    pooled = []
    for level in levels:
        rows = sweep.summarize([row for row in episodes if row["noise"] == level], by_path=True)
        assert len(rows) == len(SPEEDS) * len(DELAYS) * len(ROADS) * len(sweep.CONTROLLERS)
        cells.extend(dict(noise=level, **row) for row in rows)
        for controller in sweep.CONTROLLERS:
            group = [row for row in rows if row["controller"] == controller]
            pooled.append(dict(noise=level, controller=controller,
                               mean_penalized_rmse_m=mean(row["rmse_penalized_m"] for row in group),
                               first_places=sum(row["rank"] == 1 for row in group), cells=len(group),
                               completed=sum(row["completed"] for row in group),
                               episodes=sum(row["episodes"] for row in group)))
        group = pooled[-len(sweep.CONTROLLERS):]
        group.sort(key=lambda row: row["mean_penalized_rmse_m"])
        for rank, row in enumerate(group, 1):
            row["rank"] = rank
        pooled[-len(group):] = group
    return cells, pooled


def clean_episodes():
    with (HERE.parent / "fresh_episodes.csv").open() as stream:
        for row in csv.DictReader(stream):
            row["speed_kmh"] = int(row["speed_kmh"])
            row["delay_ms"] = int(row["delay_ms"])
            if row["speed_kmh"] not in SPEEDS or row["delay_ms"] not in DELAYS or row["path"] not in ROADS:
                continue
            assert int(row["physical_seed"]) in FRESH_SEEDS
            for name in ("completed", "mpc_fallbacks"):
                row[name] = int(row[name])
            for name in ("rmse_penalized_m", "rmse_observed_m", "saturation_fraction",
                         "applied_saturation_fraction"):
                row[name] = float(row[name])
            yield dict(noise="clean", **row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    load_policies()
    sweep.sweep.P["test"]["seeds"] = FRESH_SEEDS
    jobs = [
        (level, speed, delay, path, controller, seed)
        for level, speed, delay, path, controller in product(LEVELS, SPEEDS, DELAYS, ROADS, sweep.CONTROLLERS)
        for seed in (sweep.TRAIN_SEEDS if controller in ("LANS", "PPO") else (-1,))
    ]
    rows = list(clean_episodes())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(replay, job) for job in jobs]
        for idx, future in enumerate(as_completed(tasks), 1):
            rows.extend(future.result())
            if idx % 48 == 0:
                print(f"{idx}/{len(jobs)} jobs finished", flush=True)
    rows.sort(key=lambda row: (row["noise"], row["speed_kmh"], row["delay_ms"], row["path"],
                               row["controller"], int(row["policy_seed"]), float(row["amplitude"]),
                               int(row["physical_seed"])))
    assert len(rows) == 3 * len(SPEEDS) * len(DELAYS) * len(ROADS) * 80
    cells, pooled = summarize(rows)
    write("noise_episodes.csv", rows)
    write("noise_by_cell.csv", cells)
    write("noise_summary.csv", pooled)


if __name__ == "__main__":
    main()
