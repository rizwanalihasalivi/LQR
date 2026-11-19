"""Diagnostic replay of an offline steering prefix followed by the retained actor."""
import csv
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO

from constrained_expert import plan
from distill import HERE, STUDY, POLICY_SEEDS, actor, make_env
from evaluate import cases

PREFIX_LENGTHS = (0, 1, 5, 20, 40)


def main():
    predictors = [
        actor(PPO.load(HERE / "initial_trial" / f"policy_{seed}.zip", device="cpu"))
        for seed in POLICY_SEEDS
    ]
    outcomes = defaultdict(list)
    for case in cases("test"):
        if case["delay"] != "none":
            continue
        config = dict(case)
        config["seed"] = config.pop("physical_seed")
        env = make_env(config)
        env.reset(seed=config["seed"])
        optimal = plan(env, .00005, .0001)
        for predict in predictors:
            for cutoff in PREFIX_LENGTHS:
                obs, _ = env.reset(seed=config["seed"])
                squared_error = float(env.x[0])**2
                for tick in range(1000):
                    action = optimal[tick] + env.K @ env.x if tick < cutoff else predict(obs)
                    obs, _, terminated, truncated, info = env.step([action])
                    squared_error += info["error"]**2
                    if terminated or truncated:
                        break
                squared_error += (999-tick) * max(2.5, abs(info["error"]))**2
                key = case["path"], case["mass"], cutoff
                outcomes[key].append((np.sqrt(squared_error / 1001), truncated and not terminated))

    with (STUDY / "results" / "rank_objective_matched.csv").open() as stream:
        references = {(r["path"], float(r["mass"])): r["matched_mpc_rmse"]
                      for r in csv.DictReader(stream) if r["delay"] == "none"}
    with (HERE / "nominal_handoff.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "mass", "oracle_prefix", "rmse", "mpc_rmse", "completed", "episodes"))
        for (path, mass, cutoff), episodes in sorted(outcomes.items()):
            writer.writerow((path, mass, cutoff, np.mean([rmse for rmse, _ in episodes]),
                             references[path, mass], sum(done for _, done in episodes), len(episodes)))


if __name__ == "__main__":
    main()
