"""Check a common reset command against the nominal DLC lower bounds for all masses."""
import csv
from collections import defaultdict

import numpy as np

from distill import HERE, STUDY, make_env
from batch_sim import build_cases
from calibration import SEEDS, SCALES
from evaluate import cases
from nominal_bound import PREFIX, response
from prefix_lock_diagnosis import lower_bound
from receding_nominal_targets import SteeringTeacher


def bounds_for(cases_by_seed, mpc_path):
    paired = defaultdict(dict)
    reference = defaultdict(list)
    for case in cases_by_seed:
        if case["delay"] == "none" and case["path"] == "dlc":
            key = case["seed"], case["amplitude"], case["time_scale"]
            paired[key][case["mass"]] = case
    with mpc_path.open() as stream:
        for row in csv.DictReader(stream):
            if row["path"] == "dlc" and row["delay"] == "none":
                reference[float(row["mass"])].append(float(row["rmse_penalized"]))
    bounds = defaultdict(list)
    for (seed, _, _), group in paired.items():
        nominal = make_env(group[1650.])
        nominal_obs, _ = nominal.reset(seed=seed)
        command = SteeringTeacher(nominal).action(nominal, 0)
        for mass, case in group.items():
            env = make_env(case)
            obs, _ = env.reset(seed=seed)
            if not np.array_equal(obs, nominal_obs):
                raise RuntimeError("DLC reset observations differ across tested masses")
            state = env.x.copy()
            initial = float(state[0])
            free = np.empty(PREFIX)
            for tick in range(PREFIX):
                disturbance = env.Gd * env.ref[2, tick] + env.Hd * env.ref[3, tick]
                state = env.Ad @ state + disturbance / env.vx
                free[tick] = state[0]
            matrix = response(tuple(map(tuple, env.Ad)), tuple(env.Bd))
            bounds[mass].append(lower_bound(matrix, free, initial, np.array([command]), 1))
    for mass, values in sorted(bounds.items()):
        bound = float(np.mean(values))
        mpc = float(np.mean(reference[mass]))
        yield mass, len(values), bound, mpc, bound / mpc


def main():
    benchmark = [dict({k: v for k, v in case.items() if k != "physical_seed"},
                      seed=case["physical_seed"]) for case in cases("test")]
    splits = {
        "calibration": (build_cases(SEEDS, SCALES), HERE / "calibration" / "matched_mpc.csv"),
        "benchmark": (benchmark, STUDY / "results" / "test" / "matched_mpc_-1.csv"),
    }
    with (HERE / "shared_mass_reset_bound.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "mass", "paired_resets", "rmse_lower_bound",
                         "matched_mpc_rmse", "lower_bound_ratio"))
        for split, (cases_by_seed, mpc_path) in splits.items():
            writer.writerows((split, *row) for row in bounds_for(cases_by_seed, mpc_path))


if __name__ == "__main__":
    main()
