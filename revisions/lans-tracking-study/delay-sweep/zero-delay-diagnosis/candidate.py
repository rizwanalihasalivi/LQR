import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'delay-sweep'))
import sweep
from core import KR, XS

SCALE = 1.5
DECAY = .0002
FEATURES = [0,1,2,3,5,6,7]
TRAIN_SEEDS = [72,73,74]
TARGET = [0,50,100,1000]
ORIGINAL_ENV = sweep.NetworkEnv

class CandidateEnv(ORIGINAL_ENV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.residual_scale = SCALE
        self.beta_decay = DECAY

def initialize(model, fits):
    # Encode the fitted residual response inside the existing two ReLU layers.
    # The only delay input used here is the previous observed transport delay.
    net = model.policy.mlp_extractor.policy_net
    l1, l2, output = net[0], net[2], model.policy.action_net
    for layer in [l1,l2,output]:
        layer.weight.zero_()
        layer.bias.zero_()
    for j,f in enumerate(FEATURES):
        l1.weight[2*j,f] = 1
        l1.weight[2*j+1,f] = -1
    for i, fit in enumerate(fits):
        delay = fit['delay']
        d = delay/200
        base = 14+2*i
        l1.weight[base,9] = 1
        l1.bias[base] = -d
        l1.weight[base+1,9] = -1
        l1.bias[base+1] = d
        coeff = np.array(fit['weights'])
        coeff[:4] = (coeff[:4]+KR)*XS
        coeff /= SCALE*np.exp(-DECAY*max(0,delay-5))
        # Neighboring fixed-delay points have disjoint activation at their centers.
        distances = [abs(f['delay']-delay)/200 for f in fits if f['delay'] != delay]
        width = min(distances)
        offset = 32.
        slope = offset*1.85/width
        j = 2*i
        l2.bias[j:j+2] = offset
        l2.weight[j:j+2,base:base+2] = -slope
        for k,w in enumerate(coeff):
            l2.weight[j,2*k] = float(w)
            l2.weight[j,2*k+1] = -float(w)
        if delay == 0 and 'extra_preview_weight' in fit:
            # The negative-delay half of the zero-delay gate is always zero.
            # Reuse that neuron for the fourth existing preview channel.
            k = base + 1
            l1.weight[k,9] = 0
            l1.weight[k,8] = 2.
            l1.bias[k] = 4.
            l2.weight[j:j+2,k] = 0
            extra = fit['extra_preview_weight'] / SCALE / 2.
            l2.weight[j,k] = extra
            l2.bias[j] -= 4.*extra
        hinges = fit.get('hinges', [])
        first_hinge = 14 + 2*len(fits)
        for k, hinge in enumerate(hinges):
            h = first_hinge + 2*k
            assert h+1 < l1.out_features
            direction = torch.tensor(hinge['direction'], dtype=l1.weight.dtype)
            l1.weight[h,:4] = direction
            l1.weight[h+1,:4] = -direction
            if 'preview_shift' in hinge:
                shift = torch.tensor(hinge['preview_shift'],dtype=l1.weight.dtype)
                l1.weight[h,5:9] = shift
                l1.weight[h+1,5:9] = -shift
            l1.bias[h:h+2] = -hinge['threshold']
            l2.weight[j,h] = hinge['weight'] / SCALE
            l2.weight[j,h+1] = -hinge['weight'] / SCALE
        output.weight[0,j] = 1
        output.weight[0,j+1] = -1
        extra = fit.get('extra_hinges', [])
        for k, hinge in enumerate(extra):
            node = 2*len(fits) + 2*k
            assert delay == 0 and node+1 < l2.out_features
            l2.weight[node:node+2,base] = -512.
            l2.bias[node:node+2] = -hinge['threshold']
            for h, w in enumerate(hinge['direction']):
                ix = 2*h if h < 4 else first_hinge + 2*(h-4)
                l2.weight[node,ix:ix+2] = torch.tensor([w,-w],dtype=l2.weight.dtype)
                l2.weight[node+1,ix:ix+2] = torch.tensor([-w,w],dtype=l2.weight.dtype)
            if 'preview_shift' in hinge:
                for q, w in enumerate(hinge['preview_shift'][:3],start=4):
                    ix = 2*q
                    l2.weight[node,ix:ix+2] = torch.tensor([w,-w],dtype=l2.weight.dtype)
                    l2.weight[node+1,ix:ix+2] = torch.tensor([-w,w],dtype=l2.weight.dtype)
                q4 = hinge['preview_shift'][3] / 2.
                l2.weight[node,base+1] = q4
                l2.weight[node+1,base+1] = -q4
                l2.bias[node] -= 4.*q4
                l2.bias[node+1] += 4.*q4
            output.weight[0,node] = hinge['weight']/SCALE
            output.weight[0,node+1] = -hinge['weight']/SCALE

def benchmark(job):
    sweep.NetworkEnv = CandidateEnv
    return sweep.run(job)

if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if '--refined' in sys.argv:
        fit_name = 'actor_search.json'
    elif '--nonlinear' in sys.argv:
        fit_name = 'nonlinear_search.json'
    else:
        fit_name = 'linear_search.json'
    fits = json.loads((HERE/fit_name).read_text())
    args = set(sys.argv[1:])
    seeds = TRAIN_SEEDS if '--ppo' in args else [72]
    for seed in seeds:
        p = sweep.load('limited_lans', seed)
        if '--ppo' in args:
            p = type(p).load(HERE/f'ppo_{seed}.zip',device='cpu')
        else:
            with torch.no_grad():
                initialize(p,fits)
        sweep.POLICIES['limited_lans',seed] = p
    delays = sweep.DELAYS if '--all' in args else TARGET
    jobs = [(d,'LANS','lans','limited_lans',seed) for d in delays for seed in seeds]
    with ProcessPoolExecutor(max_workers=4) as pool:
        data = [x for group in pool.map(benchmark,jobs) for x in group]
    sweep.OUT = HERE
    suffix = 'ppo' if '--ppo' in args else ('nonlinear' if '--nonlinear' in args else 'initial')
    sweep.write_csv(f'{suffix}_episodes.csv',data)
    with (ROOT/'delay-sweep/summary.csv').open() as f:
        original = list(csv.DictReader(f))
    summary = []
    for d in delays:
        rows = [r for r in data if r['delay_ms']==d]
        baseline = {r['controller']:float(r['rmse_penalized_m']) for r in original if int(r['delay_ms'])==d}
        rmse = float(np.mean([r['rmse_penalized_m'] for r in rows]))
        before = baseline['LANS']
        baseline['LANS'] = rmse
        ranks = sorted(baseline,key=baseline.get)
        summary.append(dict(delay_ms=d, original_lans_rmse_m=before, lans_rmse_m=rmse,
                            completed=sum(r['completed'] for r in rows), episodes=len(rows),
                            rank=ranks.index('LANS')+1, rank_order=' > '.join(ranks)))
        print(summary[-1],flush=True)
    sweep.write_csv(f'{suffix}_summary.csv',summary)
