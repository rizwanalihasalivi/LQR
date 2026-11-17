"""Replay the declared development-only teacher grid, with legacy numerics.

The grid predates the precision adapter, so reproduce its recorded arithmetic
rather than silently replacing historical scores with a new implementation.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import importlib.util
import json
from precise_features import LEGACY

spec = importlib.util.spec_from_file_location('legacy_teacher_batch', LEGACY/'batch_sim.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)
DATA = batch.prepare(batch.build_cases([31000,31001,31002],[(.94,1.03),(1.06,.97)]))


def run(params):
    score,ok = batch.rollout(DATA,params)
    return dict(params=params,complete=int(ok.sum()),total=len(ok),
                mean=float(score.mean()),scores=score.tolist())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results/teacher_search.json')
    p.add_argument('--workers',type=int,default=4)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing scores; choose a fresh --output path.')
    grid = [dict(kp=kp,kd=kd,decay=d,proxy=proxy,fourth=1.)
            for kp,kd in [(100,20),(400,40),(1000,70),(2000,100),(4000,140)]
            for d in [40.,100.,250.,600.] for proxy in [1.,3.]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run,grid))
    with args.output.open('x') as f:
        json.dump(rows,f,indent=2)
    best = min(rows,key=lambda r:(-r['complete'],r['mean']))
    print('Development-only selected teacher:',best['params'],best['complete'],best['mean'])
