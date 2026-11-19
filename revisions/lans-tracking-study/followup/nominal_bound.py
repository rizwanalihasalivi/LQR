"""Bound nominal RMSE with only an initial prefix and free terminal state."""
import csv
from collections import defaultdict
from functools import lru_cache

import numpy as np
from scipy.optimize import lsq_linear

from distill import HERE, STUDY, make_env
from evaluate import cases

PREFIX = 40


@lru_cache
def response(transition, steering):
    a = np.array(transition)
    b = np.array(steering)
    impulse = []
    for _ in range(PREFIX):
        impulse.append(b[0])
        b = a @ b
    matrix = np.zeros((PREFIX, PREFIX))
    for column in range(PREFIX):
        matrix[column:, column] = impulse[:PREFIX-column]
    return matrix


def prefix_bound(env):
    matrix = response(tuple(map(tuple, env.Ad)), tuple(env.Bd))
    free = np.empty(PREFIX)
    state = env.x.copy()
    for tick in range(PREFIX):
        state = (
            env.Ad @ state
            + (env.Gd * env.ref[2, tick] + env.Hd * env.ref[3, tick]) / env.vx
        )
        free[tick] = state[0]
    fit = lsq_linear(matrix, -free, bounds=(-.5, .5), method="bvls", tol=1e-10, max_iter=250)
    if not np.isfinite(fit.x).all():
        raise RuntimeError(fit.message)
    # The tangent to the convex objective gives a lower bound even if the
    # numerical optimum has a residual gradient.
    error = matrix @ fit.x + free
    gradient = 2 * matrix.T @ error
    tangent_min = gradient @ (-.5 * np.sign(gradient) - fit.x)
    sse = env.x[0]**2 + max(0., error @ error + tangent_min)
    return np.sqrt(sse / 1001)


def main():
    bounds = defaultdict(list)
    for case in cases("test"):
        if case["delay"] != "none":
            continue
        case["seed"] = case.pop("physical_seed")
        env = make_env(case)
        env.reset(seed=case["seed"])
        bounds[case["path"], case["mass"]].append(prefix_bound(env))
    with (STUDY / "results" / "rank_objective_matched.csv").open() as stream:
        references = list(csv.DictReader(stream))
    with (HERE / "nominal_prefix_lower_bound.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "mass", "prefix_updates", "mean_rmse_lower_bound", "matched_mpc_rmse"))
        for row in references:
            if row["delay"] == "none":
                key = row["path"], float(row["mass"])
                writer.writerow((*key, PREFIX, np.mean(bounds[key]), row["matched_mpc_rmse"]))


if __name__ == "__main__":
    main()
