"""Fit road-tracking weights in three existing actor units on separate seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
import json
import subprocess
import tempfile

import numpy as np
from scipy.optimize import minimize
from stable_baselines3 import PPO
import torch

from benchmark import HERE, RECOVERY, SWEEP, configure, sweep
from fit import dataset, FLOAT_ARRAY, TRANSFORM, DIRECTIONS, THRESHOLDS, KR
import fit

TARGETS = ((90, 200, 10), (150, 1000, 54), (200, 0, 46))
EXTRA_TARGETS = ((15, 1000, 30), (30, 200, 32), (60, 200, 36), (200, 1000, 58))
TRAIN_SCALES = ((.893, 1.031), (1.106, .967))
VALID_SCALES = ((.909, 1.025), (1.087, .979))
NORMALIZATION = np.array([.1, .5, .05, .2, 1., 1., 1.])


def authority(delay):
    return 1.5 * np.exp(-.0002 * max(0, delay - 5))


def raw_weights(model, node, delay):
    layer = model.policy.mlp_extractor.policy_net[2]
    output = model.policy.action_net
    row = layer.weight[node].detach().numpy()
    weights = np.r_[row[:14:2] / NORMALIZATION, row[36:64:2]]
    weights = weights * authority(delay) * float(output.weight[0, node].detach())
    weights[:4] -= KR
    return weights


def update(model, node, delay, weights):
    layer = model.policy.mlp_extractor.policy_net[2]
    output = model.policy.action_net
    values = np.asarray(weights).copy()
    values[:4] += KR
    values /= authority(delay) * float(output.weight[0, node].detach())
    values[:7] *= NORMALIZATION
    # Keep the original speed/latency activation terms in both units.
    curvature_gate = {4: -2500./5., 5: -2500./6.} if node == 46 else {}
    for i, value in enumerate(values[:7]):
        gate = curvature_gate.get(i, 0.)
        layer.weight[node, 2*i] = float(value) + gate
        layer.weight[node, 2*i+1] = -float(value) + gate
    for i, value in enumerate(values[7:]):
        layer.weight[node, 36+2*i] = float(value)
        layer.weight[node, 37+2*i] = -float(value)


def fit_target(task):
    speed, delay, node, library = task
    torch.set_num_threads(1)
    model = PPO.load(RECOVERY / "switching_policy_72.zip", device="cpu")
    fit.configure_env = configure
    start = raw_weights(model, node, delay)
    if node == 46:
        start[4:6] += 2500. * authority(delay) / np.array([5., 6.])
    elif delay:
        start = start[:7]
    paths = ("step", "dlc", "straight", "curve")
    if not delay:
        paths += ("sine",)
    fit.PATHS = paths
    train = dataset(speed, delay, TRAIN_SCALES, range(51300, 51305), model, vary_resets=True)
    valid = dataset(speed, delay, VALID_SCALES, range(51900, 51907), model, vary_resets=True)
    sim = ctypes.CDLL(library).replay
    sim.argtypes = ([ctypes.c_int]*5 + [ctypes.c_double, FLOAT_ARRAY, ctypes.c_void_p]
                    + [FLOAT_ARRAY]*11 + [ctypes.c_void_p])
    sim.restype = None
    if (speed, delay, node) in EXTRA_TARGETS:
        weights_by_path = [1., 1., 4., 2.]
    elif not delay:
        weights_by_path = [1.] * len(paths)
    else:
        weights_by_path = [3., 1., 1., 1.] if speed == 90 else [1., 3., 1., 1.]

    def score(p, data, steps=None):
        count, a, b, fields = data
        rows = np.empty((len(fields[2]), 3))
        sim(len(rows), count, count if steps is None else steps, delay//10, len(p)-7,
            authority(delay), p, None, a, b, KR, *fields, TRANSFORM, DIRECTIONS,
            THRESHOLDS, rows, None)
        metric = rows[:, 0] + .6*(1.-rows[:, 1])
        per_path = metric.reshape(2, len(paths), -1).mean(axis=(0, 2))
        return float(np.average(per_path, weights=weights_by_path))

    options = [start]
    if (speed, delay) == (150, 1000):
        earlier = PPO.load(RECOVERY.parent / "gn_policy_72.zip", device="cpu")
        options.append(raw_weights(earlier, node, delay)[:7])
    if not delay:
        from scipy.linalg import solve_discrete_are
        from harmonic import make_fit
        env, _, _ = configure("LANS", "curve", delay, speed, 1., 1., "fixed_road")
        for penalty in (.001, .004, .012):
            matrix = solve_discrete_are(env.Ad, env.Bd[:, None], np.diag([1000., 0., .01, 0.]), [[penalty]])
            gains = -(env.Bd @ matrix @ env.Ad)/(penalty+env.Bd @ matrix @ env.Bd)
            options.append(np.r_[gains, make_fit(env)(gains), np.zeros(14)])
    widths = np.array([.15, .12, .7, .18, .08, .28, .45]) if delay else np.r_[
        65., 1.6, 1.9, .14, .015, .04, .08, np.full(14, .65)]
    start_score = score(start, valid)
    for prefix in ((22, 65, None) if not delay else (None, None)):
        center = min(options, key=lambda p: score(p, train, prefix))
        result = minimize(lambda x: score(center+x*widths, train, prefix),
                          np.zeros(len(center)), method="Powell",
                          bounds=[(-1., 1.)]*len(center),
                          options={"maxfev": 1200, "xtol": 1e-5, "ftol": 1e-9})
        options.append(center + result.x*widths)
    selected = min(options, key=lambda p: score(p, valid))
    result = dict(speed=speed, delay=delay, node=node, weights=selected.tolist(),
                  validation_before=start_score, validation_after=score(selected, valid))
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    targets = EXTRA_TARGETS if args.extended else TARGETS
    prefix = "extended" if args.extended else "road"
    with tempfile.TemporaryDirectory() as temporary:
        library = f"{temporary}/rollout.so"
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                        str(SWEEP / "low-speed" / "rollout.cpp"), "-o", library], check=True)
        with ProcessPoolExecutor(max_workers=3) as pool:
            records = list(pool.map(fit_target, [(*target, library) for target in targets]))
    (HERE / f"{prefix}_fit.json").write_text(json.dumps(records, indent=2) + "\n")
    for seed in sweep.TRAIN_SEEDS:
        model = PPO.load(RECOVERY / f"switching_policy_{seed}.zip", device="cpu")
        with torch.no_grad():
            for record in records:
                update(model, record["node"], record["delay"], record["weights"])
        model.save(HERE / f"{prefix}_policy_{seed}.zip")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(RECOVERY.parent))
    main()
