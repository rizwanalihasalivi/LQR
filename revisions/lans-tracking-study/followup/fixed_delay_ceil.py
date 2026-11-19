"""Offline feasibility probe with an initial saturated command and a fixed delay."""
import csv
from collections import defaultdict

import numpy as np

from constrained_expert import plan
from core import beta
from distill import HERE, STUDY, make_env
from evaluate import cases


def fixed_commands(config):
    env = make_env(config)
    obs, _ = env.reset(seed=config["seed"])
    first = -.5 * np.sign(obs[0])
    delay = 20
    for _ in range(delay):
        env.step([0.])
    return plan(env, .00005, .0001, start=delay, first=first)


def probe(config):
    env = make_env(config)
    delay = 20
    commands = fixed_commands(config)
    expected = np.r_[np.zeros(delay), commands]
    env.reset(seed=config["seed"])
    sse = float(env.x[0])**2
    worst_tracking_error = 0.
    for tick in range(1000):
        command = commands[tick] if tick < len(commands) else 0.
        action = (command + env.K @ env.x) / beta(200)
        _, _, term, trunc, info = env.step([action])
        sse += info["error"]**2
        worst_tracking_error = max(worst_tracking_error, abs(info["applied"] - expected[tick]))
        if term or trunc:
            break
    sse += (999-tick) * max(2.5, abs(info["error"]))**2
    return np.sqrt(sse/1001), trunc and not term, worst_tracking_error


def main():
    outcomes = defaultdict(list)
    for case in cases("test"):
        if case["delay"] != "fixed200":
            continue
        case["seed"] = case.pop("physical_seed")
        outcomes[case["path"]].append(probe(case))
    with (STUDY / "results" / "rank_objective_matched.csv").open() as stream:
        references = {r["path"]: min(float(r[k + "_rmse"]) for k in
                      ("limited_ppo", "fast_lqr", "matched_mpc"))
                      for r in csv.DictReader(stream) if r["delay"] == "fixed200"}
    with (HERE / "fixed_delay_oracle_recovery.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "offline_rmse", "best_baseline_rmse", "completed",
                         "episodes", "max_applied_steering_error"))
        for path, records in sorted(outcomes.items()):
            rmse, complete, error = zip(*records)
            writer.writerow((path, np.mean(rmse), references[path], sum(complete),
                             len(rmse), max(error)))


if __name__ == "__main__":
    main()
