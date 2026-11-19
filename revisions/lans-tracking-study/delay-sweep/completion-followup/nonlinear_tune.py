"""Fit spare second-layer units to preview features at the existing fixed-delay gates."""
import ctypes
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json

import numpy as np
from scipy.optimize import differential_evolution, minimize
from stable_baselines3 import PPO
import torch

from tune import A, B, KR, DECAY, SCALE, HERE, PARENT, make_data, vector

DELAYS = (1000, 1500, 2000)
TRAIN = ((.85, 1.043), (1.15, .955), (.93, .98), (1.08, 1.026))
VALID = ((.873, 1.05), (1.123, .95), (.988, 1.006))


def augment(data, features):
    q = data[1]
    extra = [np.maximum(q[:, :, j] * sign / scale - threshold, 0.)
             for j, sign, scale, threshold in features]
    return (data[0], np.ascontiguousarray(np.concatenate((q, np.stack(extra, axis=2)), axis=2)), *data[2:])


def fit(task):
    delay, branch = task
    torch.set_num_threads(1)
    model = PPO.load(PARENT / "ppo_72.zip", device="cpu")
    train = make_data(delay, TRAIN, range(33310, 33318), model)
    valid = make_data(delay, VALID, range(33340, 33348), model)
    q = train[1]
    dimensions = [float(v) for v in np.maximum(np.quantile(abs(q), .94, axis=(0, 1)), 1e-5)]
    threshold = .20 if branch == 0 else .48
    features = [(j, sign, dimensions[j], threshold) for j, sign in product(range(3), (-1., 1.))]
    train = augment(train, features)
    valid = augment(valid, features)
    oldest = {r["delay"]: r["weights"] for r in json.loads((HERE / "selected_fit.json").read_text())}
    baseline = np.r_[oldest[delay], np.zeros(6)]
    authority = SCALE * np.exp(-DECAY * max(0, delay - 5))
    evaluate = ctypes.CDLL(str(HERE / "rollout.so")).evaluate
    evaluate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, *([vector] * 11)]
    evaluate.restype = None

    def measures(p, data):
        out = np.empty((len(data[2]), 3))
        evaluate(len(out), delay//10, data[1].shape[-1], authority,
                 np.ascontiguousarray(p), A, B, KR, *data, baseline, baseline, out)
        return out

    def key(p):
        rmse, _, complete = measures(p, valid).T
        return -int(complete.sum()), float(rmse.mean())

    def objective(p):
        rmse, steps, complete = measures(p, train).T
        return float(rmse.mean() + 3.2*(1 - complete.mean()) + .25*(1 - steps.mean()/1000))

    widths = np.array([.01, .10, 1.5, .2, .15, .65, 1.3, *([.22]*6)])
    bounds = list(zip(baseline - widths, baseline + widths))
    trials = [baseline]
    local = minimize(objective, baseline, method="Nelder-Mead", bounds=bounds,
                     options={"maxiter": 1800, "xatol": 1e-7})
    trials.append(local.x)
    rng = np.random.default_rng(99600 + delay + branch)
    population = np.clip(local.x + rng.normal(size=(104, len(widths)))*widths*.22,
                         baseline-widths, baseline+widths)
    result = differential_evolution(objective, bounds, seed=66300 + delay + branch, x0=local.x,
                                    init=population, maxiter=170, tol=1e-6, polish=False)
    trials.append(result.x)
    polished = minimize(objective, result.x, method="Nelder-Mead", bounds=bounds,
                        options={"maxiter": 1500, "xatol": 1e-7})
    trials.append(polished.x)
    best = min(trials, key=key)
    rec = dict(delay=delay, branch=branch, weights=best[:7].tolist(), features=features,
               feature_weights=best[7:].tolist(), validation=[-key(best)[0], len(valid[2]), key(best)[1]],
               source_validation=[-key(baseline)[0], len(valid[2]), key(baseline)[1]])
    print(rec, flush=True)
    return rec


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        trials = list(pool.map(fit, product(DELAYS, (0, 1))))
    winners = [min((r for r in trials if r["delay"] == d),
                   key=lambda r: (-r["validation"][0], r["validation"][2])) for d in DELAYS]
    (HERE / "nonlinear_fit_trials.json").write_text(json.dumps(trials, indent=2) + "\n")
    (HERE / "nonlinear_selected_fit.json").write_text(json.dumps(winners, indent=2) + "\n")
