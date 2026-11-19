"""Measure the no-delay error of saved actors by phase on frozen cases."""
import csv
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO

from distill import HERE, STUDY, POLICY_SEEDS, actor, make_env
from evaluate import cases

PHASES = ((0, 10), (10, 40), (40, 1000))
CANDIDATES = {
    "retained": ("anchor_closed_loop", 24),
    "nominal_oracle": ("direct_fifo_nominal_polish", 48),
}


def metrics(errors, initial):
    squared = np.square(errors)
    return [np.sqrt((initial**2 + squared.sum()) / 1001),
            *[np.sum(squared[a:b]) for a, b in PHASES]]


def main():
    predictors = {
        label: [actor(PPO.load(HERE / name / f"policy_{seed}_{step}.zip", device="cpu"))
                for seed in POLICY_SEEDS]
        for label, (name, step) in CANDIDATES.items()
    }
    rows = defaultdict(list)
    for case in cases("test"):
        if case["delay"] != "none":
            continue
        seed = case.pop("physical_seed")
        key = (case["path"], case["mass"])
        reference = (
            f"matched_mpc_-1_{seed}_{case['path']}_none_m{case['mass']:g}"
            f"_a{case['amplitude']:g}_t{case['time_scale']:g}.npz"
        )
        with np.load(STUDY / "results" / "test" / "traces" / reference) as trace:
            rows["matched_mpc", *key].append(metrics(trace["trace"][:, 1], trace["initial_error"]))
        env = make_env({**case, "seed": seed})
        for label, actors in predictors.items():
            for predict in actors:
                obs, _ = env.reset(seed=seed)
                initial = float(env.x[0])
                errors = []
                for _ in range(1000):
                    obs, _, terminated, truncated, info = env.step([predict(obs)])
                    errors.append(info["error"])
                    if terminated or truncated:
                        break
                if terminated or len(errors) != 1000:
                    raise RuntimeError(f"Unexpected incomplete nominal episode for {label}, {key}")
                rows[label, *key].append(metrics(errors, initial))
    with (HERE / "nominal_phase_errors.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("policy", "path", "mass", "episodes", "mean_rmse",
                         "updates_1_10_sse", "updates_11_40_sse", "updates_41_1000_sse"))
        for key, values in sorted(rows.items()):
            writer.writerow((*key, len(values), *np.mean(values, axis=0)))


if __name__ == "__main__":
    main()
