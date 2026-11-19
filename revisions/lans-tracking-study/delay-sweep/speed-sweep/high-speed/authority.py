"""Compute full-preview actuator feasibility under the fixed road and FIFO."""
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run import configure_env, sweep

SPEEDS = (120, 150, 200)
DELAYS = (0, 200, 1000)


def evaluate(task):
    speed, delay, path, (amplitude, scale), seed = task
    env, horizon, ref = configure_env("LANS", path, delay, speed, amplitude, scale, "fixed_road")
    env.reset(seed=seed)
    env.ref = ref
    lag = delay // 10
    for tick in range(lag):
        _, _, failed, _, info = env.step([0.])
        assert info["applied"] == 0.
        if failed:
            return speed, delay, path, amplitude, scale, seed, "before_first_command", tick + 1
    eq = lil_matrix((4*horizon, 5*horizon))
    for tick in range(horizon):
        row = 4*tick
        eq[row:row+4, row:row+4] = np.eye(4)
        eq[row:row+4, 4*horizon+tick] = -env.Bd[:, None]
        if tick:
            eq[row:row+4, row-4:row] = -env.Ad
    env.reset(seed=seed)
    force = (ref[2,:horizon,None]*env.Gd + ref[3,:horizon,None]*env.Hd)/env.vx
    force[0] += env.Ad @ env.x
    bounds = [(-2.5,2.5), (None,None), (-.5,.5), (None,None)] * horizon
    bounds += [(0.,0.)]*lag + [(-.5,.5)]*(horizon-lag)
    result = linprog(np.zeros(5*horizon), A_eq=eq.tocsr(),b_eq=force.ravel(),
                     bounds=bounds, method="highs")
    if result.status not in (0,2):
        raise RuntimeError(result.message)
    if result.success:
        assert np.max(np.abs(eq@result.x-force.ravel())) < 1e-7
    return speed, delay, path, amplitude, scale, seed, "feasible" if result.success else "infeasible", ""


if __name__=="__main__":
    tasks=product(SPEEDS,DELAYS,sweep.P["test"]["families"],sweep.P["test"]["scales"],sweep.P["test"]["seeds"])
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(evaluate,tasks))
    with (HERE/"authority.csv").open("w",newline="") as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(("speed_kmh","delay_ms","path","amplitude","time_scale_at_90","physical_seed","status","first_failure_step"))
        writer.writerows(rows)
    for speed,delay in product(SPEEDS,DELAYS):
        group=[r for r in rows if r[:2]==(speed,delay)]
        print(speed,delay,[(s,sum(r[6]==s for r in group)) for s in ("feasible","infeasible","before_first_command")],flush=True)
