"""Tune the existing delay-conditioned residual coefficients on disjoint cases."""
import argparse
import ctypes
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import differential_evolution, minimize
from stable_baselines3 import PPO
import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "actor-refinement"
sys.path.insert(0, str(PARENT))
from candidate import CandidateEnv, DECAY, SCALE
from core import KR
from dynamic_env import plant
from features import T

DELAYS = (400, 1000, 1500, 2000)
FITS = {fit["delay"]: fit["weights"] for fit in json.loads((PARENT / "extended_search.json").read_text())}
PARAM_LIMIT = np.array([.22, .22, 3.5, .38, .30, 2.2, 2.8])
PATHS = ("dlc", "step", "sine", "slalom")
TRAIN_SCALES = ((.86, 1.04), (1.14, .96), (.94, 1.01), (1.07, .985))
VALID_SCALES = ((.88, 1.048), (1.12, .951), (1., 1.003))
TRAIN_SEEDS = range(33210, 33218)
VALID_SEEDS = range(33240, 33247)
A, B, G, H = map(np.ascontiguousarray, plant())
vector = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


def make_data(delay, scales, seeds, model, dynamic_preview=False):
    motion, preview, initial, first = [], [], [], []
    ticks = np.arange(1000)
    h = .01 * np.arange(1, 5)
    paths = PATHS if delay == 400 else PATHS[:3]
    for (amplitude, time_scale), path, seed in itertools.product(scales, paths, seeds):
        env = CandidateEnv(kind="lans", delay_mode=f"fixed{delay}", path=path,
                           amplitude=amplitude, time_scale=time_scale)
        obs, _ = env.reset(seed=seed)
        ref = env.ref
        first.append(env.compose(model.predict(obs, deterministic=True)[0], delay)[0])
        initial.append(env.x.copy())
        motion.append((ref[2, :1000, None]*G + ref[3, :1000, None]*H)/25)
        future = ref[0, ticks[:, None] + np.arange(1, 5)]
        if dynamic_preview:
            preview.append(np.column_stack((ref[0, :1000], ref[1, :1000], future)))
        else:
            q = (ref[0, :1000, None] + ref[1, :1000, None]*h - future)/(10*h*h)
            preview.append((q @ T)[:, :3])
    return tuple(np.ascontiguousarray(arr, dtype=np.float64) for arr in (motion, preview, initial, first))


def fit(task):
    delay, branch, settings = task
    exact = settings == "rounded"
    robust = settings != "linear"
    torch.set_num_threads(1)
    model = PPO.load(PARENT / "ppo_72.zip", device="cpu")
    train_scales = TRAIN_SCALES
    valid_scales = VALID_SCALES
    train_seeds, valid_seeds = TRAIN_SEEDS, VALID_SEEDS
    if robust:
        train_scales = ((.87, 1.034), (.89, 1.046), (.88, 1.04),
                        (1.11, .954), (1.13, .966), (1.12, .96))
        valid_scales = ((.88, 1.04), (1.12, .96), (.91, 1.035), (1.09, .966))
        train_seeds, valid_seeds = range(33440, 33447), range(33460, 33469)
    if exact:
        train_seeds, valid_seeds = range(33600, 33608), range(33620, 33630)
    train = make_data(delay, train_scales, train_seeds, model, exact)
    valid = make_data(delay, valid_scales, valid_seeds, model, exact)
    simulate = ctypes.CDLL(str(HERE / "rollout.so")).evaluate
    simulate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, *([vector] * 11)]
    simulate.restype = None
    authority = SCALE * np.exp(-DECAY * max(0, delay - 5))
    original = np.array(FITS[delay])
    box = [(-float(v), float(v)) for v in PARAM_LIMIT]
    net = model.policy.action_net
    start = 2*list(FITS).index(delay)
    head = np.r_[net.weight[0, start:start+2].detach().numpy(),
                 net.bias.detach().numpy()].astype(np.float64)
    transform = np.ascontiguousarray(T[:, :3])

    def episodes(p, data):
        results = np.empty((len(data[2]), 3))
        simulate(len(results), delay//10, -6 if exact else data[1].shape[-1], authority,
                 np.ascontiguousarray(p), A, B, KR, *data, transform, head, results)
        return results

    def objective(p):
        rmse, length, complete = episodes(p, train).T
        penalty = (1 - complete).mean()
        loss = rmse.mean() + (4. if branch == 0 else 1.5)*penalty + .3*(1 - length.mean()/1000)
        if robust:
            scale_loss = (rmse + 3*(1 - complete)).reshape(len(train_scales), -1).mean(axis=1)
            loss += .35 * scale_loss.max()
        return loss

    def measure(p, data):
        rmse, _, complete = episodes(p, data).T
        return [int(complete.sum()), len(complete), float(rmse.mean())]

    def key(p):
        count, _, error = measure(p, valid)
        return -count, error

    trials = [original]
    if robust:
        prior = json.loads((HERE / "selected_fit.json").read_text())
        trials.append(np.array(next(r["weights"] for r in prior if r["delay"] == delay)))
    if exact:
        prior = json.loads((HERE / "robust_selected_fit.json").read_text())
        trials.append(np.array(next(r["weights"] for r in prior if r["delay"] == delay)))
    if branch == 0:
        result = minimize(objective, min(trials, key=objective), method="Nelder-Mead", bounds=box,
                          options={"maxiter": 1800, "xatol": 1e-7})
        trials.append(result.x)
        # Search around the source as well as from the local completion optimum.
        rng = np.random.default_rng(4411 + delay)
        start = np.clip(result.x + rng.normal(size=(112, 7))*PARAM_LIMIT*.13, -PARAM_LIMIT, PARAM_LIMIT)
    else:
        start = "latinhypercube"
    result = differential_evolution(objective, box, seed=7440 + delay + branch,
                                    x0=min(trials, key=objective), init=start,
                                    maxiter=90 if exact else 220, popsize=16, tol=1e-6, polish=False)
    trials.append(result.x)
    polish = minimize(objective, result.x, method="Nelder-Mead", bounds=box,
                      options={"maxiter": 1800, "xatol": 1e-7})
    trials.append(polish.x)
    best = min(trials, key=key)
    rec = dict(delay=delay, branch=branch, weights=best.tolist(),
               train=measure(best, train), valid=measure(best, valid), original_valid=measure(original, valid))
    print(rec, flush=True)
    return rec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("linear", "robust", "rounded"), default="linear")
    args = parser.parse_args()
    delays = DELAYS if args.variant == "linear" else DELAYS[1:]
    branches = (0,) if args.variant == "rounded" else (0, 1)
    with ProcessPoolExecutor(max_workers=4 if args.variant == "linear" else 3) as pool:
        records = list(pool.map(fit, itertools.product(delays, branches, (args.variant,))))
    chosen = [min((r for r in records if r["delay"] == d),
                  key=lambda r: (-r["valid"][0], r["valid"][2])) for d in delays]
    prefix = "" if args.variant == "linear" else f"{args.variant}_"
    (HERE / f"{prefix}fit_trials.json").write_text(json.dumps(records, indent=2) + "\n")
    (HERE / f"{prefix}selected_fit.json").write_text(json.dumps(chosen, indent=2) + "\n")
