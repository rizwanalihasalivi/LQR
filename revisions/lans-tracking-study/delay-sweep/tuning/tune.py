import csv
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from evaluate import NetworkEnv, load
from features import T
from core import KR
from dynamic_env import HORIZON, plant

OUT = Path(__file__).resolve().parent
DELAYS = np.array([0, 50, 100, 150, 200, 300, 400, 700, 1000, 1500, 2000])
SEEDS = [72, 73, 74]
PARAMS = [(1., 1., .002)] + list(itertools.product(
    [.85, 1., 1.2], [.85, 1., 1.15], [.0015, .002, .003, .005]))
AD, BD, GD, HD = plant()
H = .01 * np.arange(1, 5)
MODELS = []
CASES = []
for delay, path, (amp, ts), seed in itertools.product(
        DELAYS, ('dlc', 'step', 'sine', 'slalom'),
        [(1., 1.)], [22000, 22001]):
    env = NetworkEnv(kind='lans', path=path, delay_mode=f'fixed{delay}',
                     amplitude=amp, time_scale=ts)
    env.reset(seed=seed)
    CASES.append((delay, env.x, env.ref))
D = np.array([c[0] for c in CASES])
X = np.array([c[1] for c in CASES])
REF = np.array([c[2] for c in CASES])
INDICES = np.arange(len(D))

def gain(q_multiplier):
    if q_multiplier == 1:
        return KR.copy()
    p = json.loads((ROOT.parent / 'lans-robust-fallback/results/robust_gain.json').read_text())['chosen']
    q = np.asarray(p['Q'])
    r = np.atleast_2d(p['R'])
    q[0, 0] *= q_multiplier
    b = BD.reshape(-1, 1)
    s = solve_discrete_are(AD, b, q, r)
    return np.linalg.solve(r + b.T @ s @ b, b.T @ s @ AD).ravel()

def rollout(params):
    q_mult, scale, decay = params
    k_gain = gain(q_mult)
    mod = scale * np.exp(-decay * np.maximum(0., D - 5))
    scores, completed = [], []
    for layers in MODELS:
        x = X.copy()
        active = np.ones(len(D), dtype=bool)
        steps = np.zeros(len(D), dtype=int)
        sse = x[:, 0] ** 2
        commands = np.zeros((HORIZON, len(D)))
        for tick in range(HORIZON):
            lateral, _, heading, _ = x.T
            current = REF[:, 0, tick] + lateral
            yaw = REF[:, 1, tick] / 25 + heading
            preview = current[:, None] + 25 * yaw[:, None] * H - REF[:, 0, tick + np.arange(1, 5)]
            q = (preview.astype(np.float32) - lateral[:, None] - 25 * heading[:, None] * H) / (10 * H**2)
            conditioned = (q.astype(np.float32) @ T).astype(np.float32)
            delay_obs = (D / 200).astype(np.float32) if tick else np.zeros(len(D))
            obs = np.column_stack((x / [.1, .5, .05, .2], np.ones(len(D)), conditioned,
                                   delay_obs, delay_obs)).astype(np.float32)
            a = obs
            for w, b in layers[:-1]:
                a = np.maximum(a @ w + b, 0.)
            w, b = layers[-1]
            res = np.clip((a @ w + b).ravel(), -1., 1.)
            commands[tick] = np.clip(-x @ k_gain + mod * res, -.5, .5)
            source = tick - D // 10
            applied = np.where(source >= 0, commands[np.maximum(0, source), INDICES], 0.)
            next_x = x @ AD.T + applied[:, None] * BD + REF[:, 2, tick, None] / 25 * GD + REF[:, 3, tick, None] / 25 * HD
            x[active] = next_x[active]
            steps[active] += 1
            sse[active] += x[active, 0] ** 2
            active &= (abs(x[:, 0]) <= 2.5) & (abs(x[:, 2]) <= .5)
            if not active.any():
                break
        pen = np.sqrt((sse + (HORIZON - steps) * np.maximum(2.5, abs(x[:, 0]))**2) / (HORIZON + 1))
        scores.append(pen.reshape(len(DELAYS), -1).mean(axis=1))
        completed.append(active.reshape(len(DELAYS), -1).sum(axis=1))
    return np.mean(scores, axis=0), np.sum(completed, axis=0)

if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    for seed in SEEDS:
        p = load('limited_lans', seed).policy
        MODELS.append([(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                       for m in [*p.mlp_extractor.policy_net, p.action_net]
                       if isinstance(m, torch.nn.Linear)])
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(rollout, PARAMS))
    baseline = results[0][0]
    losses = [float(np.mean(np.log(scores / baseline))) for scores, _ in results]
    best = min(range(len(losses)), key=losses.__getitem__)
    with (OUT / 'validation.csv').open('w') as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(['q_multiplier', 'residual_scale', 'beta_decay', 'delay_ms', 'rmse_m', 'completed_of_24'])
        for params, (scores, completed) in zip(PARAMS, results):
            w.writerows([*params, delay, score, count] for delay, score, count in zip(DELAYS, scores, completed))
    (OUT / 'selected.json').write_text(json.dumps(dict(params=PARAMS[best], gain=gain(PARAMS[best][0]).tolist(), validation_log_ratio=losses[best]), indent=2))
    print('Selected', PARAMS[best], losses[best], flush=True)
