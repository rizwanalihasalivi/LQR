"""Compute a nominal RMSE lower bound conditional on the retained first command."""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import lsq_linear
from stable_baselines3 import PPO

from distill import HERE, STUDY, POLICY_SEEDS, actor, make_env
from evaluate import cases
from shared_initial_action import prefix_data


def main():
    sweep = "--sweep" in sys.argv
    checkpoint = "initial_trial" if sweep else "broad_trial"
    fractions = (0., .25, .5, .75, .9, .95, .975, 1.) if sweep else (0.,)
    policies = [actor(PPO.load(HERE / checkpoint / f"policy_{s}.zip", device="cpu"))
                for s in POLICY_SEEDS]
    scores = defaultdict(list)
    for case in cases("test"):
        if case["delay"] != "none":
            continue
        case["seed"] = case.pop("physical_seed")
        obs, matrix, free = prefix_data(case)
        env = make_env(case)
        env.reset(seed=case["seed"])
        later = matrix[:, 1:]
        initial_error = float(env.x[0])
        oracle_first = 0.
        if sweep:
            oracle = lsq_linear(matrix, -free, bounds=(-.5, .5), method="bvls",
                                tol=1e-10, max_iter=250)
            oracle_first = oracle.x[0]
        for policy in policies:
            _, _, _, _, info = env.step([policy(obs)])
            retained = info["u"]
            initial, _ = env.reset(seed=case["seed"])
            assert np.array_equal(initial, obs)
            for fraction in fractions:
                command = retained + fraction * (oracle_first - retained)
                offset = free + matrix[:, 0] * command
                fit = lsq_linear(later, -offset, bounds=(-.5, .5), method="bvls",
                                 tol=1e-10, max_iter=250)
                error = later @ fit.x + offset
                gradient = 2 * later.T @ error
                tangent_min = gradient @ (-.5 * np.sign(gradient) - fit.x)
                sse = initial_error**2 + max(0., error @ error + tangent_min)
                scores[case["path"], case["mass"], fraction].append(np.sqrt(sse / 1001))

    with (STUDY / "results" / "rank_objective_matched.csv").open() as stream:
        references = list(csv.DictReader(stream))
    name = "first_command_amplitude_bound.csv" if sweep else "fixed_first_command_bound.csv"
    with (HERE / name).open("w") as stream:
        writer = csv.writer(stream)
        columns = ("path", "mass", "mean_rmse_lower_bound", "matched_mpc_rmse")
        writer.writerow((*columns, "fraction_to_oracle") if sweep else columns)
        for row in references:
            if row["delay"] == "none":
                key = row["path"], float(row["mass"])
                for fraction in fractions:
                    values = (*key, np.mean(scores[*key, fraction]), row["matched_mpc_rmse"])
                    writer.writerow((*values, fraction) if sweep else values)



if __name__ == "__main__":
    main()
