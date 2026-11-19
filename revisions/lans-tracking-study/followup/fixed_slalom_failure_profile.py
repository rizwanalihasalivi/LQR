"""Compare fast 200 ms slalom failure timing after the early steering update."""
import csv

import numpy as np
from stable_baselines3 import PPO

from distill import HERE, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases
from calibration import SEEDS, SCALES

SOURCES = (("joint", "gauss_newton_joint", 12),
           ("prefix_150", "anchor_prefix_focus", 150),
           ("prefix_450", "anchor_prefix_focus", 450),
           ("prefix_900", "anchor_prefix_focus", 900),
           ("repair_900", "repair_fixed_slalom_900", 20))


def main():
    rows = []
    for label, folder, update in SOURCES:
        for seed in POLICY_SEEDS:
            policy = PPO.load(HERE / folder / f"policy_{seed}_{update}.zip", device="cpu")
            predict = actor(policy)
            for case in build_cases(SEEDS, [SCALES[-1]]):
                if (case["path"], case["delay"]) != ("slalom", "fixed200"):
                    continue
                env = make_env(case)
                obs, _ = env.reset(seed=case["seed"])
                peak = 0.
                saturated = 0
                first = 0.
                for tick in range(1000):
                    obs, _, terminated, truncated, info = env.step([predict(obs)])
                    if tick == 0:
                        first = info["u"]
                    peak = max(peak, abs(info["error"]))
                    saturated += int(abs(info["u"]) >= .4999)
                    if terminated or truncated:
                        break
                rows.append((label, seed, case["seed"], tick+1, truncated and not terminated,
                             first, saturated, peak))
    with (HERE / "fixed_slalom_failure_profile.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("policy", "policy_seed", "physical_seed", "steps", "completed",
                         "first_steering_rad", "saturated_updates", "peak_lateral_error_m"))
        writer.writerows(rows)


if __name__ == "__main__":
    main()
