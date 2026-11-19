"""Fit existing-policy weights for three speeds on disjoint rollout cases."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import solve_discrete_are
from stable_baselines3 import PPO
import torch

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from run import configure_env, sweep
from candidate import SCALE, DECAY
from core import KR
from features import T

SPEEDS = (15, 30, 60)
DELAYS = (0, 200, 1000)
FITS = {r["delay"]: r for r in json.loads((SWEEP / "actor-refinement" / "extended_search.json").read_text())}
HINGES = FITS[0]["hinges"]
DIRECTIONS = np.array([h["direction"] for h in HINGES])
THRESHOLDS = np.array([h["threshold"] for h in HINGES])
TRANSFORM = np.ascontiguousarray(T[:, :3])
PATHS = ("dlc", "step", "sine", "slalom")
FLOAT_ARRAY = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
TRAIN = ((.89, 1.038), (1.105, .961), (.98, 1.008))
VALID = ((.875, 1.045), (1.124, .958))
TRAIN_SEEDS = (34100, 34101, 34102, 34103)
VALID_SEEDS = (34140, 34141, 34142, 34143, 34144, 34145)


def dataset(speed, delay, scales, seeds, model, *, limit=None, vary_resets=False):
    data, poses, initial, at_reset = [], [], [], []
    for scale_index, (amplitude, scale) in enumerate(scales):
        for path_index, path in enumerate(PATHS):
            env, count, ref = configure_env("LANS", path, delay, speed, amplitude, scale, "fixed_road")
            if limit is not None:
                count = min(count, limit)
            index = np.arange(count)
            vx = np.full(count, env.vx)
            reference = np.column_stack((ref[0, :count], ref[1, :count]/vx, vx,
                                         ref[0, index[:, None] + np.arange(1, 5)]))
            disturbance = (env.Gd * ref[2, :count, None] + env.Hd * ref[3, :count, None])/env.vx
            for seed in seeds:
                if vary_resets:
                    seed += 100*(4*scale_index+path_index)
                env.reset(seed=seed)
                env.ref = ref
                action = model.predict(env.obs(), deterministic=True)[0]
                at_reset.append(env.compose(action, delay)[0])
                data.append(disturbance)
                poses.append(reference)
                initial.append(env.x.copy())
    return count, np.ascontiguousarray(env.Ad), np.ascontiguousarray(env.Bd), tuple(map(np.ascontiguousarray, (data, poses, initial, at_reset)))


def load_simulator():
    library = ctypes.CDLL(str(HERE / "rollout.so"))
    function = library.replay
    function.argtypes = ([ctypes.c_int]*5 + [ctypes.c_double, FLOAT_ARRAY, ctypes.c_void_p]
                         + [FLOAT_ARRAY]*11 + [ctypes.c_void_p])
    function.restype = None
    return function


def prepare(speed, delay):
    torch.set_num_threads(1)
    model = PPO.load(SWEEP / "completion-followup" / "final_policy_72.zip", device="cpu")
    sim = load_simulator()
    train = dataset(speed, delay, TRAIN, TRAIN_SEEDS, model)
    valid = dataset(speed, delay, VALID, VALID_SEEDS, model)
    authority = SCALE * np.exp(-DECAY*max(0, delay-5))
    def evaluate(p, data, steps=None, startup=None, tail=None):
        count, a, b, arrays = data
        out = np.empty((len(arrays[2]), 3))
        prefix = None if startup is None else startup.ctypes.data
        asymptotic = None if tail is None else tail.ctypes.data
        sim(len(out), count, count if steps is None else steps, delay//10, len(p)-7, authority,
            np.ascontiguousarray(p), prefix, a, b, KR, *arrays, TRANSFORM, DIRECTIONS, THRESHOLDS,
            out, asymptotic)
        return out
    return evaluate, train, valid


def stat(rows):
    return [float(rows[:, 0].mean()), int(rows[:, 1].sum()), len(rows)]


def nominal(speed):
    evaluate, train, valid = prepare(speed, 0)
    original = np.r_[FITS[0]["weights"], [h["weight"] for h in HINGES]]
    width = np.r_[24., 2., 2.5, .25, .65, .4, .6, np.full(14, 2.)]
    bounds = list(zip(original-width, original+width))
    def objective(p):
        # The initial correction dominates nominal RMSE on these courses.
        return evaluate(p, train, steps=85)[:, 0].mean()
    trials = [original]
    for mode in (0, 1):
        initial = trials[-1].copy()
        if mode:
            _, a, b, _ = train
            q = np.diag([1000., 0., .003, 0.])
            mat = solve_discrete_are(a, b[:, None], q, np.array([[3e-5]]))
            initial[:4] = np.clip(-(b @ mat @ a)/(3e-5+b @ mat @ b),
                                  original[:4]-width[:4], original[:4]+width[:4])
        fit = minimize(objective, initial, method="Nelder-Mead", bounds=bounds,
                       options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-12})
        trials.append(fit.x)
    # Use full-route independent error for choice, including the steady-state tracking tail.
    best = min(trials, key=lambda p: evaluate(p, valid)[:, 0].mean())
    record = dict(speed=speed, delay=0, weights=best.tolist(), train=stat(evaluate(best, train)),
                  validation=stat(evaluate(best, valid)), baseline=stat(evaluate(original, valid)))
    print(record, flush=True)
    return record


def delayed(task):
    speed, delay = task
    evaluate, train, valid = prepare(speed, delay)
    fit = json.loads((HERE / "nominal_fit.json").read_text())
    startup = np.array(next(r["weights"] for r in fit if r["speed"] == speed))
    original = np.array(FITS[delay]["weights"])
    bounds = list(zip(*np.array([[-1.25, -.5, -2.8, -.8, -1.0, -2.4, -3.0],
                                [  .15,  .4,  2.8,  .8,  1.0,  2.4,  3.0]])))
    # Probe weak feedback and the unchanged LQR as well as the source residual.
    candidates = [original, np.r_[-KR, 0., 0., 0.],
                  np.r_[-.25*KR, original[4:]], np.r_[-.1*KR, 0., 0., 0.]]
    lo, hi = np.array(bounds).T
    candidates = [np.clip(x,lo,hi) for x in candidates]
    def objective(p):
        out = evaluate(p, train, startup=startup)
        return out[:, 0].mean() + 1.1*(1-out[:, 1].mean())
    best = min(candidates, key=objective)
    rng = np.random.default_rng(6628 + speed + delay)
    scales = np.array([.04,.02,.16,.04,.10,.25,.32])
    nearby = np.clip(best + rng.normal(size=(52,7))*scales, lo, hi)
    initial = np.vstack((nearby, rng.uniform(lo,hi,size=(12,7))))
    result = differential_evolution(objective, bounds, x0=best, init=initial, seed=8901+speed+delay,
                                    maxiter=115, tol=1e-5, polish=False)
    candidates.append(result.x)
    for guess in (best, result.x):
        result = minimize(objective, guess, bounds=bounds, method="Nelder-Mead",
                          options={"maxiter": 750, "xatol": 2e-6})
        candidates.append(result.x)
    best = min(candidates, key=lambda p: objective_on(p, evaluate, valid, startup))
    record = dict(speed=speed, delay=delay, weights=best.tolist(),
                  train=stat(evaluate(best, train, startup=startup)),
                  validation=stat(evaluate(best, valid, startup=startup)),
                  baseline=stat(evaluate(original, valid)))
    print(record, flush=True)
    return record


def objective_on(weights, evaluate, cases, startup):
    out = evaluate(weights, cases, startup=startup)
    return out[:, 0].mean()+1.1*(1-out[:, 1].mean())


def polish(speed):
    evaluate, _, _ = prepare(speed, 0)
    model = PPO.load(SWEEP / "completion-followup" / "final_policy_72.zip", device="cpu")
    early = dataset(speed, 0, TRAIN, range(38000, 38009), model, limit=110, vary_resets=True)
    late = dataset(speed, 0, ((.9,1.025), (1.09,.974)), range(39310,39312), model)
    valid = dataset(speed, 0, ((.86,1.043), (1.13,.962)), range(39340,39348), model, vary_resets=True)
    original = np.r_[FITS[0]["weights"], [h["weight"] for h in HINGES]]
    started = json.loads((HERE / "nominal_fit.json").read_text())
    previous = np.array(next(r["weights"] for r in started if r["speed"]==speed))
    trials = [original, previous]
    width = np.r_[25., 1.2, 2., .22, .6, .4, .4, np.full(14,1.4)]
    box = list(zip(original-width, original+width))

    def startup_error(p):
        return evaluate(p, early)[:,0].mean()

    for start in (original, previous):
        result = minimize(startup_error, start, method="Nelder-Mead", bounds=box,
                          options={"maxiter": 2200, "xatol": 1e-6, "fatol": 1e-12})
        trials.append(result.x)

    def tune_feedforward(p):
        def steady_error(features):
            candidate = p.copy()
            candidate[4:7] = features
            return evaluate(candidate, late)[:,0].mean()
        result = minimize(steady_error, p[4:7], method="Nelder-Mead", bounds=box[4:7],
                          options={"maxiter": 120, "xatol": 1e-6, "fatol": 1e-12})
        refined = p.copy()
        refined[4:7] = result.x
        return refined

    trials.extend([tune_feedforward(p) for p in trials])
    best = min(trials, key=lambda p:evaluate(p, valid)[:,0].mean())
    record = dict(speed=speed, delay=0, weights=best.tolist(),
                  validation=stat(evaluate(best, valid)), baseline=stat(evaluate(original, valid)),
                  previous=stat(evaluate(previous, valid)))
    print(record, flush=True)
    return record


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("nominal", "delayed", "polish"))
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=4) as pool:
        if args.stage == "nominal":
            records = list(pool.map(nominal, SPEEDS))
        elif args.stage == "delayed":
            records = list(pool.map(delayed, product(SPEEDS, DELAYS[1:])))
        else:
            records = list(pool.map(polish, SPEEDS))
    filename = "nominal_polish.json" if args.stage == "polish" else f"{args.stage}_fit.json"
    (HERE/filename).write_text(json.dumps(records, indent=2)+"\n")
