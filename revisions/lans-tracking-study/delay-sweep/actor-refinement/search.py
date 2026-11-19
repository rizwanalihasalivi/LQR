import ctypes
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import differential_evolution, minimize

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from evaluate import NetworkEnv
from core import KR, XS, teacher
from features import T, unwhiten
from dynamic_env import plant

A, B, G, H = map(np.ascontiguousarray, plant())
B = B.reshape(4)
lib = ctypes.CDLL(str(HERE / 'rollout.so'))
vec = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
lib.rollout.argtypes = [ctypes.c_int, ctypes.c_int, *([vec] * 8)]
lib.rollout.restype = None
teacher_params = json.loads((ROOT / 'results/teacher_search.json').read_text())['selected']['parameters']
DELAYS = [0, 50, 100, 150, 200, 300, 400, 700, 1000, 1500, 2000]

def make_data(scales, seeds):
    disturbance, preview, initial = [], [], []
    step = np.arange(1000)
    h = .01 * np.arange(1, 5)
    for (amp, ts), path, seed in itertools.product(scales, ['dlc','step','sine','slalom'], seeds):
        e = NetworkEnv(kind='lans', path=path, amplitude=amp, time_scale=ts)
        e.reset(seed=seed)
        r = e.ref
        initial.append(e.x.copy())
        disturbance.append((r[2,:1000,None]*G+r[3,:1000,None]*H)/25)
        q = (r[0,:1000,None] + r[1,:1000,None]*h -
             r[0,step[:,None]+np.arange(1,5)])/(10*h*h)
        preview.append((q@T)[:,:3])
    return tuple(np.ascontiguousarray(x, dtype=np.float64) for x in (disturbance,preview,initial))

TRAIN = make_data([(.82,.93),(1,1),(1.18,1.11)],[88112,88113,88114])
VALID = make_data([(.94,1.02),(1.06,.98)],[22000,22001])

def metrics(d,p,zero,data):
    out = np.empty((len(data[-1]), 3))
    lib.rollout(len(out), d//10, np.ascontiguousarray(p), np.ascontiguousarray(zero),
                A, B, *data, out)
    return out.mean(0)

def fit(task):
    d, zero = task
    # Current model-informed initialization and low-gain delayed alternatives.
    w = np.array([(teacher(unwhiten(np.eye(11)[j]*.01),teacher_params)-
                  teacher(np.zeros(11),teacher_params))/.01 for j in range(9)])
    start = np.r_[w[:4]/XS,w[5:8]]
    limits = np.array([max(abs(v)*3, m) for v,m in zip(start,[.2,.15,2,.3,2.,1.,1.])])
    b = B[:,None]
    R = .0002
    s = solve_discrete_are(A, b, np.diag([1000.,0,.001,0]), np.array([[R]]))
    dare = -np.linalg.solve(R+b.T@s@b,b.T@s@A).ravel()
    if d == 0:
        limits[:4] = np.maximum(limits[:4], abs(dare)*2)
    else:
        limits[:4] = np.minimum(limits[:4], np.array([2.,.7,12.,1.]))
    if d >= 300:
        limits[:4] = [.20,.10,1.6,.22]
    candidates = [np.clip(start,-limits,limits), np.r_[-KR,start[4:]],
                  np.zeros(7), np.r_[np.zeros(4),start[4:]]]
    if d == 0:
        candidates.append(np.r_[dare,start[4:]])
    z = zero if zero is not None else np.zeros(7)
    def loss(p,data=TRAIN):
        rmse, steps, complete = metrics(d,p,z,data)
        return rmse + (0.025*(1-complete)+.015*(1-steps/1000) if d >= 300 else 0)
    bounds = list(zip(-limits,limits))
    initial = min(candidates,key=loss)
    trials = candidates.copy()
    for seed in range(3):
        result = differential_evolution(loss,bounds,x0=initial,seed=8123+seed,
                                        maxiter=120,popsize=9,tol=1e-5,polish=False)
        initial = result.x
        trials.append(result.x)
        refined = minimize(loss,result.x,method='Nelder-Mead',bounds=bounds,
                           options={'maxiter':1100,'xatol':1e-8,'fatol':1e-9})
        trials.append(refined.x)
    best = min(trials,key=lambda p:loss(p,VALID))
    res = dict(delay=d, weights=best.tolist(), train=metrics(d,best,z,TRAIN).tolist(),
               validation=metrics(d,best,z,VALID).tolist())
    print(res,flush=True)
    return res

if __name__ == '__main__':
    zero = fit((0,None))
    with ProcessPoolExecutor(max_workers=4) as pool:
        target = [50,100,1000,150,200,300,400,700,1500,2000]
        results = [zero,*pool.map(fit,[(d,np.array(zero['weights'])) for d in target])]
    (HERE/'linear_search.json').write_text(json.dumps(results,indent=2))
