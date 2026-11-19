import csv
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent
sys.path.insert(0, str(UPSTREAM))
import sweep

CHOICE = json.loads((HERE / 'selected.json').read_text())
OriginalEnv = sweep.NetworkEnv

class TunedEnv(OriginalEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.K = np.array(CHOICE['gain'])
        _, self.residual_scale, self.beta_decay = CHOICE['params']

def check(job):
    sweep.NetworkEnv = TunedEnv
    return sweep.run(job)

if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sweep.OUT = HERE
    for seed in [72, 73, 74]:
        sweep.POLICIES['limited_lans', seed] = sweep.load('limited_lans', seed)
    jobs = [(delay, 'LANS', 'lans', 'limited_lans', seed)
            for delay, seed in itertools.product(sweep.DELAYS, [72, 73, 74])]
    with ProcessPoolExecutor(max_workers=4) as pool:
        episodes = []
        for index, batch in enumerate(pool.map(check, jobs), 1):
            episodes.extend(batch)
            if index % 3 == 0:
                print(f"Scalar evaluation completed for {batch[0]['delay_ms']} ms", flush=True)
    assert len(episodes) == 1320
    sweep.write_csv('selected_episodes.csv', episodes)
    with (UPSTREAM / 'summary.csv').open() as stream:
        old = list(csv.DictReader(stream))
    summary = []
    for delay in sweep.DELAYS:
        previous = [r for r in old if int(r['delay_ms']) == delay]
        assert len(previous) == 4
        cases = [r for r in episodes if r['delay_ms'] == delay]
        assert len(cases) == 120
        ranks = {r['controller']: float(r['rmse_penalized_m']) for r in previous}
        original = ranks['LANS']
        ranks['LANS'] = mean(r['rmse_penalized_m'] for r in cases)
        ranked = sorted(ranks, key=ranks.get)
        summary.append(dict(delay_ms=delay, original_lans_rmse_m=original,
                            tuned_lans_rmse_m=ranks['LANS'],
                            completed=sum(r['completed'] for r in cases),
                            tuned_lans_rank=ranked.index('LANS') + 1,
                            rank_order=' > '.join(ranked)))
    sweep.write_csv('comparison.csv', summary)
    print(f"Rank 1 at {sum(r['tuned_lans_rank']==1 for r in summary)}/{len(summary)} delays")
