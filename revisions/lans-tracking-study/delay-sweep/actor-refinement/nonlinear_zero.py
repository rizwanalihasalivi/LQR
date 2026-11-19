import json
from pathlib import Path

import numpy as np
import torch
from search import A, B, make_data, XS

HERE = Path(__file__).resolve().parent
FITS = json.loads((HERE/'linear_search.json').read_text())
BASE = np.array(FITS[0]['weights'])
# Odd hinges of lateral and lateral-rate feedback let the existing network
# behave differently when steering saturates. All use present observations.
directions = []
thresholds = []
for direction, knots in [
    ([1., .15, 0, 0], [.035, .10, .25, .55]),
    ([0, 1, 0, 0], [.025,.10,.30]),
    ([1., .50, 0, 0], [.05,.15,.40]),
    ([1., -.55, 0, 0], [.06,.22]),
    ([0, 0, 1., 0], [.12,.35]),
]:
    for knot in knots:
        directions.append(direction)
        thresholds.append(knot)

def data(scales,seeds):
    return tuple(torch.tensor(x) for x in make_data(scales,seeds))

def loss(weights, data, horizon, regularizer=0.):
    disturb, preview, x = data
    matrix = A_T
    sse = x[:,0].square()
    for t in range(horizon):
        z = (x/X_SCALE) @ DIR_T
        hinge = (z-THRESH).relu()-(-z-THRESH).relu()
        u = (x @ BASE_T[:4] + preview[:,t] @ BASE_T[4:] + hinge @ weights).clamp(-.5,.5)
        x = x @ matrix + u[:,None]*B_T + disturb[:,t]
        sse = sse + x[:,0].square()
    return (sse/1001).sqrt().mean() + regularizer*weights.square().mean()

A_T = torch.tensor(A.T.copy())
B_T = torch.tensor(B.copy())
X_SCALE = torch.tensor(XS)
DIR_T = torch.tensor(np.array(directions).T.copy())
THRESH = torch.tensor(thresholds,dtype=torch.float64)
BASE_T = torch.tensor(BASE)

if __name__ == '__main__':
    torch.set_num_threads(1)
    train = data([(.87,.97),(1.12,1.06)],list(range(88340,88366)))
    valid = data([(.92,1.02),(1.08,.98)],list(range(22310,22325)))
    weights = torch.zeros(len(thresholds),dtype=torch.float64,requires_grad=True)
    zero = weights.detach().clone()
    with torch.no_grad():
        best = float(loss(zero,valid,1000))
    best_weights = zero
    opt = torch.optim.Adam([weights],lr=.025)
    for epoch in range(360):
        opt.zero_grad()
        score = loss(weights,train,80,2e-7)
        score.backward()
        torch.nn.utils.clip_grad_norm_([weights],.001)
        opt.step()
        if (epoch+1)%30==0:
            with torch.no_grad():
                v = float(loss(weights,valid,1000))
                if v < best:
                    best = v
                    best_weights = weights.detach().clone()
            print(epoch+1, float(score.detach()), v, best,flush=True)
    with torch.no_grad():
        print('Baseline',float(loss(zero,valid,1000)),'chosen',best,flush=True)
    FITS[0]['hinges'] = [dict(direction=d,threshold=t,weight=float(w))
                        for d,t,w in zip(directions,thresholds,best_weights)]
    (HERE/'nonlinear_search.json').write_text(json.dumps(FITS,indent=2))
