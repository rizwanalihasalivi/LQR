"""Offline QP labels for mixed delay under the original FIFO."""
import csv
from collections import defaultdict

import numpy as np
from scipy import sparse

from calibration import SEEDS, SCALES
from batch_sim import build_cases
from constrained_expert import plan
from core import beta
from distill import HERE, make_env, STUDY


def fifo_map(delays):
    ticks = np.arange(len(delays))
    releases = np.maximum.accumulate(ticks + np.ceil(delays / 10).astype(int))
    generations = np.searchsorted(releases, ticks, side="right") - 1
    active = generations >= 0
    schedule = sparse.csc_matrix(
        (np.ones(active.sum()), (ticks[active], generations[active])),
        shape=(len(ticks), len(ticks)),
    )
    return schedule, generations


def commands(env):
    schedule, generations = fifo_map(env.delays[:1000])
    return plan(env, .00005, .0001, schedule=schedule), generations


def main():
    outcomes = defaultdict(list)
    for case in build_cases(SEEDS, SCALES):
        if case["delay"] != "mixed":
            continue
        env = make_env(case)
        env.reset(seed=case["seed"])
        planned, generations = commands(env)
        sse = float(env.x[0]) ** 2
        max_error = 0.
        for tick, command in enumerate(planned):
            action = (command + env.K @ env.x) / beta(env.delays[tick])
            _, _, terminated, truncated, info = env.step([action])
            expected = planned[generations[tick]] if generations[tick] >= 0 else 0.
            max_error = max(max_error, abs(info["applied"] - expected))
            sse += info["error"] ** 2
            if terminated or truncated:
                break
        sse += (999-tick) * max(2.5, abs(info["error"])) ** 2
        outcomes[case["path"]].append((np.sqrt(sse/1001), truncated and not terminated, max_error))

    references = defaultdict(list)
    with (HERE / "calibration" / "limited_ppo.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["delay"] == "mixed":
                references[row["path"]].append(float(row["rmse_penalized"]))
    with (HERE / "mixed_offline_expert.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "offline_rmse", "frozen_ppo_rmse", "completed",
                         "episodes", "max_applied_steering_error"))
        for path, records in sorted(outcomes.items()):
            rmse, completed, error = zip(*records)
            writer.writerow((path, np.mean(rmse), np.mean(references[path]),
                             sum(completed), len(records), max(error)))


if __name__ == "__main__":
    main()
