"""Offline-only nominal steering over a full reference with bounded actuation."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import osqp
from scipy import sparse

from distill import make_env
from calibration import SEEDS, SCALES
from batch_sim import build_cases


def plan(env, input_weight=.001, rate_weight=.01, heading_weight=.01, *, start=0, first=None, schedule=None):
    """Solve the fixed initialization and known reference as a sparse QP."""
    a, b = env.Ad, env.Bd
    horizon = 1000 - start
    identity = sparse.eye(horizon, format="csc")
    shift = sparse.diags([np.ones(horizon - 1)], [-1], format="csc")
    dynamics = sparse.kron(identity, np.eye(4)) - sparse.kron(shift, a)
    command_map = identity if schedule is None else schedule
    controls = -sparse.kron(command_map, b[:, None])
    equality = sparse.hstack([dynamics, controls], format="csc")
    lower = (env.Gd[:, None] * env.ref[2, start:1000] + env.Hd[:, None] * env.ref[3, start:1000]).T / env.vx
    lower[0] += a @ env.x
    lower = lower.ravel()
    bounds = sparse.hstack([sparse.csc_matrix((horizon, 4*horizon)), identity], format="csc")
    rate = identity - shift
    steering_cost = input_weight * identity + rate_weight * (rate.T @ rate)
    state_cost = sparse.kron(identity, np.diag([1000., 0., heading_weight, 0.]))
    weight = sparse.block_diag((state_cost, steering_cost), format="csc")
    steering_lower = np.full(horizon, -.5)
    steering_upper = np.full(horizon, .5)
    if first is not None:
        steering_lower[0] = first
        steering_upper[0] = first
    qp = osqp.OSQP()
    qp.setup(
        P=2*weight, q=np.zeros(5*horizon),
        A=sparse.vstack([equality, bounds], format="csc"),
        l=np.r_[lower, steering_lower],
        u=np.r_[lower, steering_upper],
        eps_abs=1e-7, eps_rel=1e-7, max_iter=10000, polishing=True, verbose=False,
    )
    result = qp.solve(raise_error=True)
    return np.clip(result.x[-horizon:], -.5, .5)


if __name__ == "__main__":
    benchmark = "--benchmark" in sys.argv
    if benchmark:
        from evaluate import cases
        source = [
            dict({k: v for k, v in c.items() if k != "physical_seed"}, seed=c["physical_seed"])
            for c in cases("test")
        ]
        weights = [(.00005, .0001)]
        label = "benchmark"
    else:
        source = build_cases(SEEDS[:3], SCALES)
        weights = [(.001, .01), (.00005, .0001), (.0002, .002)]
        label = "calibration"
    results = defaultdict(list)
    for case in source:
        if case["delay"] != "none":
            continue
        for input_weight, rate_weight in weights:
            env = make_env(case)
            env.reset(seed=case["seed"])
            commands = plan(env, input_weight, rate_weight)
            sse = float(env.x[0])**2
            for tick, command in enumerate(commands):
                _, _, terminated, _, info = env.step([command + env.K @ env.x])
                sse += info["error"]**2
                if terminated:
                    break
            sse += (999-tick)*max(2.5, abs(info["error"]))**2
            results[case["path"], case["mass"], input_weight, rate_weight].append(np.sqrt(sse/1001))
    with (Path(__file__).parent / f"constrained_oracle_{label}.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(("path", "mass", "input_weight", "rate_weight", "rmse"))
        for key, scores in sorted(results.items()):
            writer.writerow((*key, np.mean(scores)))
