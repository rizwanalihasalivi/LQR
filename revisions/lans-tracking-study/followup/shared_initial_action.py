"""Measure the first-steering constraint shared across nominal DLC masses."""
import csv

import numpy as np
from scipy.optimize import lsq_linear

from distill import HERE, make_env
from evaluate import cases
from nominal_bound import PREFIX, response


def prefix_data(case):
    env = make_env(case)
    obs, _ = env.reset(seed=case["seed"])
    matrix = response(tuple(map(tuple, env.Ad)), tuple(env.Bd))
    state = env.x.copy()
    free = []
    for tick in range(PREFIX):
        state = env.Ad @ state + (env.Gd * env.ref[2, tick] + env.Hd * env.ref[3, tick]) / env.vx
        free.append(state[0])
    return obs, matrix, np.array(free)


def main():
    groups = {}
    for case in cases("test"):
        if case["path"] == "dlc" and case["delay"] == "none":
            case["seed"] = case.pop("physical_seed")
            key = case["seed"], case["amplitude"], case["time_scale"]
            groups.setdefault(key, {})[case["mass"]] = prefix_data(case)
    masses = (1100., 1650., 2200.)
    rows = []
    for key, group in sorted(groups.items()):
        assert all(np.array_equal(group[m][0], group[masses[0]][0]) for m in masses)
        joint = np.zeros((3*PREFIX, 1 + 3*(PREFIX-1)))
        error = np.concatenate([group[m][2] for m in masses])
        individual = []
        for i, mass in enumerate(masses):
            _, matrix, free = group[mass]
            result = lsq_linear(matrix, -free, bounds=(-.5, .5), method="bvls", tol=1e-10, max_iter=250)
            assert result.success
            individual.append(result)
            block = slice(i*PREFIX, (i+1)*PREFIX)
            joint[block, 0] = matrix[:, 0]
            joint[block, 1+i*(PREFIX-1):1+(i+1)*(PREFIX-1)] = matrix[:, 1:]
        result = lsq_linear(joint, -error, bounds=(-.5, .5), method="bvls", tol=1e-10, max_iter=250)
        assert result.success
        rows.append((*key, *(x.x[0] for x in individual), result.x[0],
                     2*(result.cost - sum(x.cost for x in individual))))
    with (HERE / "shared_initial_action.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed", "amplitude", "time_scale", "first_1100", "first_1650",
                         "first_2200", "shared_first", "extra_prefix_sse_m2"))
        writer.writerows(rows)
    print("Mean extra squared error:", np.mean([row[-1] for row in rows]))


if __name__ == "__main__":
    main()
