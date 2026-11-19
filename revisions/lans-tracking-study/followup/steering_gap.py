"""Compare fixed learned checkpoints against an offline nominal DLC steering plan."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from constrained_expert import plan
from distill import POLICY_SEEDS, actor, make_env
from evaluate import cases


def measure(checkpoints, output):
    predictors = [
        actor(PPO.load(checkpoints / f"policy_{seed}.zip", device="cpu"))
        for seed in POLICY_SEEDS
    ]
    rows = defaultdict(list)
    for case in cases("test"):
        if case["delay"] != "none" or case["path"] != "dlc":
            continue
        config = dict(case)
        config["seed"] = config.pop("physical_seed")
        env = make_env(config)
        env.reset(seed=config["seed"])
        oracle = plan(env, .00005, .0001)
        for predict in predictors:
            obs, _ = env.reset(seed=config["seed"])
            for tick in range(40):
                obs, _, terminated, _, info = env.step([predict(obs)])
                command = info["u"]
                rows[case["mass"], tick].append((abs(command), abs(command - oracle[tick])))
                if terminated:
                    break
    with output.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mass", "tick", "episodes", "mean_abs_lans_steering_rad",
                         "mean_abs_oracle_gap_rad"))
        for (mass, tick), samples in sorted(rows.items()):
            writer.writerow((mass, tick, len(samples), *np.mean(samples, axis=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    measure(args.checkpoints, args.output)
