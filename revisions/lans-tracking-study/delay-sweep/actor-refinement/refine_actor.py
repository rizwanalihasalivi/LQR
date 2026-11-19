import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from nonlinear_zero import data, A_T, B_T, X_SCALE, BASE_T

HERE = Path(__file__).resolve().parent
FITS = json.loads((HERE/'nonlinear_search.json').read_text())
H = FITS[0]['hinges']
BASE_DIRECTIONS = torch.tensor([h['direction'] for h in H],dtype=torch.float64)
BASE_THRESHOLDS = torch.tensor([h['threshold'] for h in H],dtype=torch.float64)
BASE_WEIGHTS = torch.tensor([h['weight'] for h in H],dtype=torch.float64)
TRAIN = data([(.90,.96),(1.12,1.07)],list(range(89110,89154)))
VALID = data([(.94,1.03),(1.07,.96)],list(range(22420,22446)))

def rollout(params,dataset,horizon):
    directions, thresholds, weights, linear = params
    d, q, x = dataset
    sse = x[:,0].square()
    for t in range(horizon):
        f = x/X_SCALE
        z = f @ directions.T
        h = (z-thresholds).relu()-(-z-thresholds).relu()
        u = (f @ linear + q[:,t] @ BASE_T[4:] + h @ weights).clamp(-.5,.5)
        x = x @ A_T + u[:,None]*B_T + d[:,t]
        sse = sse + x[:,0].square()
    return (sse/1001).sqrt()

def fit(mode):
    torch.set_num_threads(1)
    init = [BASE_DIRECTIONS,BASE_THRESHOLDS,BASE_WEIGHTS,BASE_T[:4]*X_SCALE]
    params = [x.detach().clone().requires_grad_() for x in init]
    optim = torch.optim.Adam([dict(params=[params[0],params[1]],lr=.003),
                              dict(params=[params[2]],lr=.012),
                              dict(params=[params[3]],lr=.004)])
    with torch.no_grad():
        baseline = float(rollout(init,VALID,1000).mean())
    best = baseline
    kept = init
    table = []
    for epoch in range(600):
        optim.zero_grad()
        rmse = rollout(params,TRAIN,80)
        initial_y = TRAIN[2][:,0].abs()
        weight = 1 + mode*torch.exp(-initial_y/.022)
        reg = 1e-8*sum((x-a).square().mean() for x,a in zip(params,init))
        objective = (weight*rmse).mean()+reg
        objective.backward()
        optim.step()
        with torch.no_grad():
            params[1].clamp_(.015,1.0)
            if (epoch+1)%30==0:
                val = float(rollout(params,VALID,1000).mean())
                table.append((epoch+1,val))
                if val < best:
                    best = val
                    kept = [p.detach().clone() for p in params]
        if (epoch+1)%150==0:
            print(mode,epoch+1,best,baseline,flush=True)
    return mode,best,[x.tolist() for x in kept],table

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:
        candidates = list(pool.map(fit,[0.,1.,3.]))
    mode, best, saved, _ = min(candidates,key=lambda c:c[1])
    directions, thresholds, weights, linear = saved
    FITS[0]['weights'][:4] = (torch.tensor(linear,dtype=torch.float64)/X_SCALE).tolist()
    FITS[0]['hinges'] = [dict(direction=d,threshold=t,weight=w)
                       for d,t,w in zip(directions,thresholds,weights)]
    (HERE/'actor_search.json').write_text(json.dumps(FITS,indent=2))
    (HERE/'actor_validation.json').write_text(json.dumps([
        dict(mode=mode,score=score,checkpoints=table) for mode,score,_,table in candidates],indent=2))
    print('chosen',mode,best,flush=True)
