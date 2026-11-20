"""Fit the existing nominal actor rows to a clean and noisy training mixture."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
from itertools import product
import json
from pathlib import Path
from subprocess import run
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
from stable_baselines3 import PPO
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from noise_replay import LEVELS, RECOVERY, ROADS, SPEEDS, configure, sweep
sys.path.insert(0, str(RECOVERY.parent))
from fit import DIRECTIONS, THRESHOLDS, TRANSFORM, KR, FLOAT_ARRAY
from harmonic import make_fit

STATE_SCALE = np.array([.1, .5, .05, .2, 1., 1., 1.])
ROWS = {90: 0, 150: 44}
LEVEL_SIGMAS = ((0.,) * 4, *LEVELS.values())
TRAIN = ((.91, 1.027), (1.095, .974))
VALID = ((.95, 1.013), (1.052, .985))
HORIZONS = np.arange(1, 5) * .01


def dataset(speed, scales, seeds):
    initial, force, preview, noise = [], [], [], []
    for (amplitude, scale), path, seed in product(scales, ROADS, seeds):
        env, ticks, road = configure("LANS", path, 0, speed, amplitude, scale, "fixed_road")
        env.reset(seed=seed)
        steps = np.arange(ticks)
        delta = (road[0, steps, None] + road[1, steps, None] * HORIZONS -
                 road[0, steps[:, None] + np.arange(1, 5)]) / (10. * HORIZONS**2)
        disturbance = (road[2, steps, None] * env.Gd + road[3, steps, None] * env.Hd) / env.vx
        draws = np.random.default_rng(np.random.SeedSequence([83017, seed])).normal(size=(ticks, 4))
        for sigma in LEVEL_SIGMAS:
            initial.append(env.x)
            force.append(disturbance)
            preview.append(delta @ TRANSFORM)
            noise.append(draws * sigma)
    return ticks, tuple(np.ascontiguousarray(x) for x in
                        (env.Ad, env.Bd, KR, initial, force, preview, noise))


def from_actor(model, speed):
    layer = model.policy.mlp_extractor.policy_net[2]
    output = model.policy.action_net
    row = ROWS[speed]
    source = layer.weight[row].detach().numpy()
    p = np.r_[source[:14:2] / STATE_SCALE, source[36:64:2]]
    p *= 1.5 * float(output.weight[0, row].detach())
    p[:4] -= KR
    return p


def to_actor(model, speed, weights):
    row = ROWS[speed]
    layer = model.policy.mlp_extractor.policy_net[2]
    output = model.policy.action_net
    p = np.array(weights)
    p[:4] += KR
    p /= 1.5 * float(output.weight[0, row].detach())
    p[:7] *= STATE_SCALE
    layer.weight[row, :14:2] = torch.as_tensor(p[:7], dtype=layer.weight.dtype)
    layer.weight[row, 1:14:2] = -layer.weight[row, :14:2]
    layer.weight[row, 36:64:2] = torch.as_tensor(p[7:], dtype=layer.weight.dtype)
    layer.weight[row, 37:64:2] = -layer.weight[row, 36:64:2]
    if speed == 150:
        # These eight terms belong to the original 150 km/h nominal fit.
        output.weight[0, 14:22].zero_()


def optimize(task):
    speed, library, variant = task
    torch.set_num_threads(1)
    model = PPO.load(RECOVERY / "switching_policy_72.zip", device="cpu")
    env, _, _ = configure("LANS", "curve", 0, speed, 1., 1., "fixed_road")
    train = dataset(speed, TRAIN, range(55100, 55105))
    validation = dataset(speed, VALID, range(56300, 56308))
    simulate = ctypes.CDLL(library).score
    simulate.argtypes = [ctypes.c_int] * 2 + [FLOAT_ARRAY] * 10 + [ctypes.c_int, FLOAT_ARRAY]
    simulate.restype = None

    def assess(params, data, use_hinges=True):
        ticks, inputs = data
        result = np.empty(len(inputs[3]))
        simulate(len(result), ticks, *inputs, np.ascontiguousarray(params),
                 DIRECTIONS, THRESHOLDS, len(DIRECTIONS) if use_hinges else 0, result)
        return result.reshape(-1, len(LEVEL_SIGMAS)).mean(axis=0)

    # Give each noise severity a comparable scale, with extra clean weight
    # to keep the gain search from trading away nominal tracking entirely.
    scales = np.array([.0035, .007, .011]) * (speed / 90)**.5
    mixture = np.array([1.15, 1., 1.3]) if variant == "noise" else np.array([.30, 1., 4.])
    metric = lambda scores: float(np.average(scores / scales, weights=mixture))
    starts = [from_actor(model, speed)]
    ff = make_fit(env)
    penalties = (.022, .11, .55, 2.75, 13.75)
    if variant == "moderate":
        penalties += (68.75, 343.75)
    for penalty in penalties:
        matrix = solve_discrete_are(env.Ad, env.Bd[:, None], np.diag([1000., 0., .015, 0.]), [[penalty]])
        gains = -(env.Bd @ matrix @ env.Ad)/(penalty + env.Bd @ matrix @ env.Bd)
        starts.append(np.r_[gains, ff(gains), np.zeros(len(DIRECTIONS))])

    ranked = sorted(starts, key=lambda p: metric(assess(p, train)))
    candidates = starts.copy()
    widths = np.array([22., 1.1, 1.9, .25, .003, .025, .12])
    for source in ranked[:2]:
        center = source[:7].copy()
        result = minimize(
            lambda x: metric(assess(center + x * widths, train, use_hinges=False)),
            np.zeros(len(widths)), method="Powell", bounds=[(-1., 1.)] * len(widths),
            options={"maxfev": 680, "xtol": 1e-4, "ftol": 5e-7},
        )
        candidates.append(np.r_[center + result.x * widths, np.zeros(len(DIRECTIONS))])
    center = min(candidates, key=lambda p: metric(assess(p, train)))
    # Refine only the eight largest-magnitude hinge features from the original row.
    selected_hinges = np.argsort(abs(starts[0][7:]))[-8:] + 7
    indices = np.r_[np.arange(4), selected_hinges]
    increments = np.r_[widths[:4] * .35, np.full(len(selected_hinges), .25)]

    def expand(x):
        candidate = center.copy()
        candidate[indices] += x * increments
        return candidate

    result = minimize(
        lambda x: metric(assess(expand(x), train)), np.zeros(len(indices)), method="Powell",
        bounds=[(-1., 1.)] * len(indices),
        options={"maxfev": 860, "xtol": 1e-4, "ftol": 3e-7},
    )
    candidates.append(expand(result.x))
    selected = min(candidates, key=lambda p: metric(assess(p, validation)))
    record = dict(speed=speed, weights=selected.tolist(), validation_m=assess(selected, validation).tolist(),
                  source_affine_and_hinges_m=assess(starts[0], validation).tolist())
    print(record, flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="noise", choices=("noise", "moderate"))
    args = parser.parse_args()
    with TemporaryDirectory() as directory:
        library = str(Path(directory) / "fit.so")
        run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(HERE / "nominal_fit.cpp"),
             "-o", library], check=True)
        with ProcessPoolExecutor(max_workers=len(SPEEDS)) as pool:
            records = list(pool.map(optimize, [(speed, library, args.variant) for speed in SPEEDS]))
    (HERE / f"{args.variant}_fit.json").write_text(json.dumps(records, indent=2) + "\n")
    for seed in sweep.TRAIN_SEEDS:
        model = PPO.load(RECOVERY / f"switching_policy_{seed}.zip", device="cpu")
        with torch.no_grad():
            for record in records:
                to_actor(model, record["speed"], record["weights"])
        model.save(HERE / f"{args.variant}_policy_{seed}.zip")


if __name__ == "__main__":
    main()
