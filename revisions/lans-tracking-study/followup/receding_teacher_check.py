"""Check offline teacher headroom on nominal calibration cases."""
import csv
from collections import defaultdict

import numpy as np

from distill import HERE, make_env
from batch_sim import build_cases
from calibration import SEEDS, SCALES
from receding_nominal_targets import SteeringTeacher


def main():
    reference, results = defaultdict(list), defaultdict(list)
    with (HERE / "calibration" / "matched_mpc.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["delay"] == "none":
                reference[row["path"], float(row["mass"])].append(float(row["rmse_penalized"]))
    for case in build_cases(SEEDS, SCALES):
        if case["delay"] != "none":
            continue
        env = make_env(case)
        env.reset(seed=case["seed"])
        teacher = SteeringTeacher(env)
        sse = float(env.x[0])**2
        for tick in range(1000):
            action = teacher.action(env, tick) + env.K @ env.x
            _, _, terminated, truncated, info = env.step([action])
            sse += info["error"]**2
            if terminated or truncated:
                break
        sse += (999 - tick) * max(2.5, abs(info["error"]))**2
        results[case["path"], case["mass"]].append((np.sqrt(sse/1001), truncated and not terminated))
    with (HERE / "receding_teacher_check.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "mass", "offline_teacher_rmse", "frozen_mpc_rmse",
                         "completed", "episodes"))
        for key, records in sorted(results.items()):
            rmse, complete = zip(*records)
            writer.writerow((*key, np.mean(rmse), np.mean(reference[key]),
                             sum(complete), len(complete)))


if __name__ == "__main__":
    main()
