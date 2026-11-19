"""Bound calibration RMSE after fixing an actor's first steering commands."""
import csv
from collections import defaultdict

import numpy as np
from scipy.optimize import lsq_linear
from stable_baselines3 import PPO

from distill import HERE, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases
from calibration import SEEDS, SCALES
from nominal_bound import PREFIX, response

LOCKS = (0, 1, 2, 3, 5, 10, 20, 40)


def lower_bound(matrix, free, initial, commands, lock):
    remainder = free + matrix[:, :lock] @ commands[:lock]
    adjustable = matrix[:, lock:]
    if adjustable.shape[1] == 0:
        return np.sqrt((initial**2 + remainder @ remainder) / 1001)
    fit = lsq_linear(adjustable, -remainder, bounds=(-.5, .5),
                     method="bvls", tol=1e-10, max_iter=250)
    error = adjustable @ fit.x + remainder
    gradient = 2 * adjustable.T @ error
    tangent = gradient @ (-.5 * np.sign(gradient) - fit.x)
    return np.sqrt((initial**2 + max(0., error @ error + tangent)) / 1001)


def main():
    sources = {
        "retained_joint": ("gauss_newton_joint", 12),
        "prefix_joint": ("anchor_prefix_focus", 150),
        "slalom_anchored_joint": ("anchor_prefix_slalom", 480),
        "nominal_precision": ("nominal_oracle_precision", 4000),
        "receding_nominal": ("receding_nominal_targets", 5000),
    }
    predictors = {
        kind: [actor(PPO.load(HERE / folder / f"policy_{seed}_{step}.zip", device="cpu"))
               for seed in POLICY_SEEDS]
        for kind, (folder, step) in sources.items()
    }
    bounds = defaultdict(list)
    mpc = defaultdict(list)
    with (HERE / "calibration" / "matched_mpc.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["delay"] == "none":
                mpc[row["path"], float(row["mass"])].append(float(row["rmse_penalized"]))
    for case in build_cases(SEEDS, SCALES):
        if case["delay"] != "none":
            continue
        env = make_env(case)
        env.reset(seed=case["seed"])
        x = env.x.copy()
        initial = float(x[0])
        matrix = response(tuple(map(tuple, env.Ad)), tuple(env.Bd))
        free = np.empty(PREFIX)
        for tick in range(PREFIX):
            x = env.Ad @ x + (env.Gd * env.ref[2, tick] + env.Hd * env.ref[3, tick]) / env.vx
            free[tick] = x[0]
        for kind, policies in predictors.items():
            for predict in policies:
                obs, _ = env.reset(seed=case["seed"])
                actions = []
                for _ in range(PREFIX):
                    obs, _, terminated, _, info = env.step([predict(obs)])
                    if terminated:
                        raise RuntimeError(f"Unexpected nominal termination for {kind}")
                    actions.append(info["applied"])
                commands = np.array(actions)
                for lock in LOCKS:
                    key = kind, case["path"], case["mass"], lock
                    bounds[key].append(lower_bound(matrix, free, initial, commands, lock))
    with (HERE / "prefix_lock_diagnosis.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("actor", "path", "mass", "locked_updates", "episodes",
                         "mean_rmse_lower_bound", "matched_mpc_rmse", "lower_bound_ratio"))
        for (kind, path, mass, limit), values in sorted(bounds.items()):
            bound = np.mean(values)
            target = np.mean(mpc[path, mass])
            writer.writerow((kind, path, mass, limit, len(values), bound, target, bound/target))


if __name__ == "__main__":
    main()
