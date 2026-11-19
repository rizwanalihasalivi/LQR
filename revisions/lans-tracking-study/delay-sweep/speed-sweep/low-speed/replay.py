"""Encode the selected coefficients in one existing LANS network and score it."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json

import numpy as np
from stable_baselines3 import PPO
import torch

from fit import DELAYS, FITS, HERE, SCALE, SPEEDS, SWEEP, DECAY
import run as speed_sweep
from batched import run as batched_run
from candidate import FEATURES, TRAIN_SEEDS
from core import KR, XS

SOURCE = SWEEP / "completion-followup"
LOW_SPEED_GATES = (50, 100, 150, 300, 400, 700)


def encode(model, records):
    """Reuse the existing speed and delay inputs and both 64-unit layers."""
    first, _, layer, _ = model.policy.mlp_extractor.policy_net
    output = model.policy.action_net
    index = {delay: i for i, delay in enumerate(FITS)}
    delay_zero = 14 + 2*index[0]
    available = [delay_zero+1]
    # With d >= 0, ReLU(d-c) = d - c + ReLU(c-d).
    layer.weight[:, delay_zero+1].zero_()
    for delay in LOW_SPEED_GATES:
        positive = 14+2*index[delay]
        column = layer.weight[:, positive].clone()
        layer.weight[:, delay_zero].add_(column)
        layer.weight[:, positive+1].add_(column)
        layer.bias.sub_(column*(delay/200))
        layer.weight[:, positive].zero_()
        available.append(positive)
    gate_nodes = {}
    for i, speed in enumerate(SPEEDS):
        left, right = available[2*i:2*i+2]
        gate_nodes[speed] = (left, right)
        for node, sign in ((left, 1.), (right, -1.)):
            first.weight[node].zero_()
            first.weight[node, 4] = sign
            first.bias[node] = -sign*(speed/90)
    cutoff = available[-1]
    first.weight[cutoff].zero_()
    first.weight[cutoff, 4] = -1.
    first.bias[cutoff] = 75/90
    for delay in DELAYS:
        k = 2*index[delay]
        layer.weight[k:k+2, cutoff] = -4000.
    for i, row in enumerate(records):
        j = 2*(len(FITS)+i)
        s1, s2 = gate_nodes[row["speed"]]
        delay = row["delay"]
        d1 = 14+2*index[delay]
        d_nodes = [d1, d1+1] if delay else [d1]
        scale = SCALE*np.exp(-DECAY*max(0,delay-5))
        coeff = np.array(row["weights"])/scale
        coeff[:4] = (coeff[:4]+KR/scale)*XS
        layer.weight[j:j+2].zero_()
        layer.weight[j:j+2, [s1,s2]] = -4000.
        layer.weight[j:j+2, d_nodes] = -1000.
        layer.bias[j:j+2] = 32.
        for k, _ in enumerate(FEATURES):
            layer.weight[j, 2*k] = float(coeff[k])
            layer.weight[j, 2*k+1] = -float(coeff[k])
        start = 14+2*len(FITS)
        for k, weight in enumerate(coeff[7:]):
            layer.weight[j, start+2*k] = float(weight)
            layer.weight[j, start+2*k+1] = -float(weight)
        output.weight[0,j] = 1.
        output.weight[0,j+1] = -1.
        if row.get("tail") is not None:
            coefficients = np.array(row["tail"][:-1])/scale
            coefficients[:4] *= XS
            extra = 2*(len(FITS)+len(records))
            layer.weight[extra:extra+2].zero_()
            layer.weight[extra:extra+2, [s1,s2]] = -4000.
            layer.weight[extra:extra+2, d_nodes] = -1000.
            layer.bias[extra:extra+2] = .25
            for k,w in enumerate(coefficients[:7]):
                layer.weight[extra, 2*k] = float(w)
                layer.weight[extra, 2*k+1] = -float(w)
            layer.weight[extra:extra+2, :2].sub_(.25*XS[0]/row["tail"][-1])
            for k,w in enumerate(coefficients[7:]):
                layer.weight[extra, start+2*k] = float(w)
                layer.weight[extra, start+2*k+1] = -float(w)
            output.weight[0,extra] = 1.
            output.weight[0,extra+1] = -1.


def configure_models(variant, build, fresh):
    records = None
    if build:
        name = {"initial":"nominal_fit.json", "polished":"nominal_polish.json",
                "precision":"nominal_precision.json", "tail":"nominal_tail.json"}[variant]
        records = json.loads((HERE / name).read_text()) + json.loads((HERE / "delayed_fit.json").read_text())
        assert {(r["speed"],r["delay"]) for r in records} == set(product(SPEEDS, DELAYS))
    for seed in TRAIN_SEEDS:
        saved = HERE / f"{variant}_policy_{seed}.zip"
        if build:
            model = PPO.load(SOURCE / f"final_policy_{seed}.zip", device="cpu")
            with torch.no_grad():
                encode(model, records)
            model.save(saved)
        else:
            model = PPO.load(saved,device="cpu")
        speed_sweep.POLICIES["LANS",seed] = model
        if fresh:
            speed_sweep.POLICIES["PPO",seed] = speed_sweep.sweep.load("limited_ppo",seed)


def original_baselines():
    with (HERE.parent / "episodes.csv").open() as stream:
        for r in csv.DictReader(stream):
            if r["profile"]!="fixed_road" or r["controller"]=="LANS":
                continue
            for k in ("speed_kmh","delay_ms","completed","mpc_fallbacks"):
                r[k]=int(r[k])
            for k in ("rmse_penalized_m","rmse_observed_m","saturation_fraction","applied_saturation_fraction"):
                r[k]=float(r[k])
            yield r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("initial","polished","precision","tail"), default="initial")
    parser.add_argument("--split", choices=("benchmark","anchor","fresh"), default="benchmark")
    parser.add_argument("--delays", type=int, nargs="+", choices=DELAYS, default=DELAYS)
    parser.add_argument("--speeds", type=int, nargs="+", choices=SPEEDS)
    parser.add_argument("--batched", action="store_true")
    args=parser.parse_args()
    fresh = args.split=="fresh"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    speed_sweep.HERE=HERE
    configure_models(args.variant, args.split=="benchmark", fresh)
    if fresh:
        speed_sweep.sweep.P["test"]={**speed_sweep.sweep.P["test"],"seeds":[34300,34301,34302]}
    speeds = [90] if args.split=="anchor" else args.speeds or SPEEDS
    controllers = speed_sweep.CONTROLLERS if fresh else ("LANS",)
    paths = speed_sweep.sweep.P["test"]["families"]
    jobs=[]
    for speed, delay, controller, path in product(speeds,args.delays,controllers,paths):
        for seed in TRAIN_SEEDS if controller in ("LANS","PPO") else (-1,):
            jobs.append(("fixed_road",speed,delay,controller,seed,path))
    records=[]
    progress={}
    prefix=f"{args.variant}_{args.split}"
    if args.speeds:
        prefix += "_" + "-".join(str(s) for s in args.speeds)
    if tuple(args.delays)!=DELAYS:
        prefix += "_delay" + "-".join(str(d) for d in args.delays)
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks={pool.submit(batched_run if args.batched else speed_sweep.run, j): j for j in jobs}
        for task in as_completed(tasks):
            group=task.result()
            records.extend(group)
            condition=tasks[task][1:3]
            progress.setdefault(condition,[]).extend(group)
            if len(progress[condition])==(192 if fresh else 120):
                subset=progress.pop(condition)
                if not fresh:
                    subset += [r for r in original_baselines() if (r["speed_kmh"],r["delay_ms"])==condition]
                row=next(r for r in speed_sweep.summarize(subset, False) if r["controller"]=="LANS")
                print(row,flush=True)
    if not fresh:
        records.extend(r for r in original_baselines() if r["speed_kmh"] in speeds
                       and r["delay_ms"] in args.delays)
    records.sort(key=lambda r:(r["speed_kmh"],r["delay_ms"],r["controller"],str(r["policy_seed"]),
                              r["path"],str(r["amplitude"]),str(r["physical_seed"])))
    speed_sweep.save(f"{prefix}_episodes.csv",records)
    speed_sweep.save(f"{prefix}_ranking.csv",speed_sweep.summarize(records,False))
    speed_sweep.save(f"{prefix}_by_path.csv",speed_sweep.summarize(records,True))


if __name__=="__main__":
    main()
