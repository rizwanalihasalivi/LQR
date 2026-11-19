import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from candidate import CandidateEnv, initialize, sweep, TRAIN_SEEDS

HERE = Path(__file__).resolve().parent

class TrainingEnv(CandidateEnv):
    def __init__(self, seed):
        super().__init__(kind='lans', delay_mode='none')
        self.training_rng = np.random.default_rng(92000+seed)

    def reset(self, *, seed=None, options=None):
        rng = self.training_rng
        self.delay_mode = f'fixed{int(rng.choice(sweep.DELAYS))}'
        self.path = str(rng.choice(['dlc','step','sine','slalom']))
        self.amplitude = float(rng.uniform(.82,1.18))
        self.time_scale = float(rng.uniform(.93,1.11))
        return super().reset(seed=int(rng.integers(93000,99000)), options=options)

class Projection(BaseCallback):
    def __init__(self, parameters, references, bound):
        super().__init__()
        self.parameters = parameters
        self.references = references
        self.bound = bound

    @torch.no_grad()
    def project(self):
        for p, r in zip(self.parameters,self.references):
            p.copy_(torch.maximum(torch.minimum(p,r+self.bound),r-self.bound))

    def _on_rollout_start(self):
        self.project()

    def _on_step(self):
        return True

def train(seed):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if '--direct' in sys.argv:
        fit_name = 'direct_fit.json'
    elif '--augmented' in sys.argv:
        fit_name = 'augmented_fit.json'
    elif '--phase' in sys.argv:
        fit_name = 'phase_fit.json'
    elif '--extra' in sys.argv:
        fit_name = 'extra_search.json'
    elif '--full' in sys.argv:
        fit_name = 'full_search.json'
    elif '--preview' in sys.argv:
        fit_name = 'preview_search.json'
    elif '--extended' in sys.argv:
        fit_name = 'extended_search.json'
    elif '--refined' in sys.argv:
        fit_name = 'actor_search.json'
    elif '--nonlinear' in sys.argv:
        fit_name = 'nonlinear_search.json'
    else:
        fit_name = 'linear_search.json'
    fits = json.loads((HERE/fit_name).read_text())
    model = sweep.load('limited_lans',seed)
    model.set_random_seed(seed)
    model.set_env(TrainingEnv(seed))
    with torch.no_grad():
        initialize(model,fits)
    for p in model.policy.mlp_extractor.policy_net.parameters():
        p.requires_grad_(False)
    model.policy.log_std.requires_grad_(False)
    parameters = list(model.policy.action_net.parameters())
    references = [p.detach().clone() for p in parameters]
    model.lr_schedule = lambda _:1e-5
    model.n_epochs = 4
    projection = Projection(parameters,references,1e-6)
    model.learn(total_timesteps=8192,callback=projection)
    projection.project()
    delta = max(float((p-r).detach().abs().max()) for p,r in zip(parameters,references))
    model.save(HERE/f'ppo_{seed}.zip')
    return dict(seed=seed,steps=model.num_timesteps,max_weight_change=delta)

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(train,TRAIN_SEEDS))
    (HERE/'finetune.json').write_text(json.dumps(records,indent=2))
    print(records,flush=True)
