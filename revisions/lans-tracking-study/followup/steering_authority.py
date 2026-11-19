"""Measure attainable delayed steering when changing only exponential decay."""
import csv
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO

from distill import HERE, POLICY_SEEDS, actor, make_env
from evaluate import cases

DECAYS = (.002, .008, .012, .025, .04)


def main():
    source = HERE / "gauss_newton_joint"
    predictors = [actor(PPO.load(source / f"policy_{seed}_12.zip", device="cpu"))
                  for seed in POLICY_SEEDS]
    observations = defaultdict(list)
    for case in cases("test"):
        if case["delay"] != "fixed200":
            continue
        case["seed"] = case.pop("physical_seed")
        env = make_env(case)
        for predict in predictors:
            obs, _ = env.reset(seed=case["seed"])
            for _ in range(1000):
                base = -float(env.K @ env.x)
                obs, _, terminated, truncated, info = env.step([predict(obs)])
                observations[case["path"]].append((base, info["u"]))
                if terminated or truncated:
                    break
    with (HERE / "fixed_delay_steering_authority.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "beta_decay_per_ms", "gain_at_200ms", "steps",
                         "unattainable_fraction", "mean_unattainable_command_gap_rad"))
        for name, samples in sorted(observations.items()):
            base, actual = np.array(samples).T
            for decay in DECAYS:
                gain = np.exp(-195 * decay)
                lower = np.clip(base - gain, -.5, .5)
                upper = np.clip(base + gain, -.5, .5)
                distance = np.maximum(lower - actual, np.maximum(actual - upper, 0.))
                writer.writerow((name, decay, gain, len(actual), np.mean(distance > 1e-8), distance.mean()))


if __name__ == "__main__":
    main()
