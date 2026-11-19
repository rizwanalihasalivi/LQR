"""Relax causality and test whether full bounded actuator authority could finish each run."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import csv
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sweep
from dynamic_env import HORIZON
from prefix_bound import DELAYS


@lru_cache(maxsize=1)
def dynamics():
    env = sweep.NetworkEnv(kind="lans")
    h = HORIZON
    row, col, val = [], [], []
    for tick in range(h):
        for i in range(4):
            r = 4*tick + i
            row.extend((r, r))
            col.extend((r, 4*h + tick))
            val.extend((1., -env.Bd[i]))
            if tick:
                row.extend([r] * 4)
                col.extend(range(4*(tick-1), 4*tick))
                val.extend(-env.Ad[i])
    equalities = coo_matrix((val, (row, col)), shape=(4*h, 5*h)).tocsr()
    return equalities


def solve(record):
    delay = int(record["delay_ms"])
    if record["terminates_before_first_command"] == "1":
        return {**record, "relaxed_feasible": 0, "status": "pre-actuation termination"}
    seed = int(record["physical_seed"])
    env = sweep.NetworkEnv(kind="lans", path=record["path"], delay_mode=f"fixed{delay}",
                           amplitude=float(record["amplitude"]), time_scale=float(record["time_scale"]))
    env.reset(seed=seed)
    disturbance = (env.ref[2, :HORIZON, None]*env.Gd + env.ref[3, :HORIZON, None]*env.Hd)/env.vx
    disturbance[0] += env.Ad @ env.x
    lag = delay // 10
    bounds = [(-2.5, 2.5), (None, None), (-.5, .5), (None, None)] * HORIZON
    bounds += [(0., 0.)] * lag + [(-.5, .5)] * (HORIZON-lag)
    result = linprog(np.zeros(5*HORIZON), A_eq=dynamics(), b_eq=disturbance.ravel(),
                     bounds=bounds, method="highs")
    if result.status not in (0, 2):
        raise RuntimeError(f"Unexpected LP status {result.status}: {result.message}")
    if result.success:
        states = result.x[:4*HORIZON].reshape(-1, 4)
        inputs = result.x[4*HORIZON:]
        x = env.x.copy()
        error = 0.
        for tick, u in enumerate(inputs):
            dd, ddd = env.ref[2:4, tick]
            x = env.Ad @ x + env.Bd*u + env.Gd*(dd/env.vx) + env.Hd*(ddd/env.vx)
            error = max(error, float(np.max(np.abs(x-states[tick]))))
        if error > 1e-6:
            raise RuntimeError(f"LP trajectory residual {error}")
    return {**record, "relaxed_feasible": int(result.success),
            "status": "feasible with future-known steering" if result.success else "infeasible"}


def main():
    with (HERE / "prefix_bound_episodes.csv").open() as stream:
        records = list(csv.DictReader(stream))
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(solve, records))
    sweep.OUT = HERE
    sweep.write_csv("authority_bound_episodes.csv", results)
    for delay in DELAYS:
        cases = [r for r in results if int(r["delay_ms"]) == delay]
        print(delay, sum(r["relaxed_feasible"] for r in cases), len(cases), flush=True)


if __name__ == "__main__":
    main()
