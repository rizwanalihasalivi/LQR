"""Save the fixed new checkpoint set and replay it in the original scalar simulator."""
import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
import sys

import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "actor-refinement"
sys.path.insert(0, str(PARENT))
from candidate import CandidateEnv, DECAY, FEATURES, ORIGINAL_ENV, SCALE, TRAIN_SEEDS, sweep
from core import XS

TARGET_DELAYS = (400, 700, 1000, 1500, 2000)
NAMES = ("LANS", "PPO", "LQR", "MPC")


def configure_models(fresh, variant):
    fits = json.loads((PARENT / "extended_search.json").read_text())
    selected = {row["delay"]: row["weights"]
                for row in json.loads((HERE / "selected_fit.json").read_text())}
    refinements = {}
    policy_prefix = "policy" if variant == "linear" else f"{variant}_policy"
    if variant == "nonlinear":
        records = json.loads((HERE / "nonlinear_selected_fit.json").read_text())
        refinements = {r["delay"]: r for r in records}
        selected.update({d: r["weights"] for d, r in refinements.items()})
    if variant in ("robust", "rounded"):
        records = json.loads((HERE / f"{variant}_selected_fit.json").read_text())
        selected.update({r["delay"]: r["weights"] for r in records})
    if variant in ("retained", "final"):
        records = json.loads((HERE / f"{variant}_fit.json").read_text())
        selected = {r["delay"]: r["weights"] for r in records}
        refinements = {r["delay"]: r for r in records if "features" in r}
    for seed in TRAIN_SEEDS:
        model = sweep.load("limited_lans", seed)
        source = type(model).load(PARENT / f"ppo_{seed}.zip", device="cpu")
        if fresh:
            sweep.POLICIES["published", seed] = source
            sweep.POLICIES["limited_ppo", seed] = sweep.load("limited_ppo", seed)
            model = type(model).load(HERE / f"{policy_prefix}_{seed}.zip", device="cpu")
        else:
            model = source
            layer = model.policy.mlp_extractor.policy_net[2]
            output = model.policy.action_net
            with torch.no_grad():
                neuron = 2*len(fits)
                for index, fit in enumerate(fits):
                    delay = fit["delay"]
                    original = fit["weights"]
                    if delay not in selected:
                        continue
                    scale = SCALE * math.exp(-DECAY * max(0, delay - 5))
                    for j, _ in enumerate(FEATURES):
                        factor = XS[j] if j < 4 else 1.
                        delta = (selected[delay][j] - original[j]) * factor / scale
                        layer.weight[2*index, 2*j].add_(delta)
                        layer.weight[2*index, 2*j+1].sub_(delta)
                    if delay in refinements:
                        extra = refinements[delay]
                        nearest = min(abs(other["delay"] - delay)/200 for other in fits
                                      if other["delay"] != delay)
                        gate = 14 + 2*index
                        for (j, sign, feature_scale, threshold), weight in zip(
                                extra["features"], extra["feature_weights"], strict=True):
                            feature = 2*(4+j)
                            layer.weight[neuron].zero_()
                            layer.bias[neuron] = -threshold
                            layer.weight[neuron, gate:gate+2] = -128./nearest
                            layer.weight[neuron, feature] = sign/feature_scale
                            layer.weight[neuron, feature+1] = -sign/feature_scale
                            output.weight[0, neuron] = weight/scale
                            neuron += 1
            model.save(HERE / f"{policy_prefix}_{seed}.zip")
        sweep.POLICIES["limited_lans", seed] = model


def run(job):
    sweep.NetworkEnv = CandidateEnv if job[2] == "lans" else ORIGINAL_ENV
    return sweep.run(job)


def summarize(candidate, references, delays):
    rows = []
    for delay in delays:
        scores = []
        for controller in NAMES:
            group = [r for r in (candidate if controller == "LANS" else references)
                     if r["delay_ms"] == delay and r["controller"] == controller]
            scores.append((mean(float(r["rmse_penalized_m"]) for r in group), controller,
                           sum(int(r["completed"]) for r in group), len(group)))
        for rank, (rmse, name, completed, runs) in enumerate(sorted(scores), 1):
            rows.append(dict(delay_ms=delay, controller=name, rank=rank, rmse_penalized_m=rmse,
                             completed=completed, episodes=runs))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--variant", choices=("linear", "nonlinear", "robust", "rounded", "retained", "final"),
                        default="linear")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sweep.OUT = HERE
    configure_models(args.fresh, args.variant)
    if args.variant in ("robust", "rounded"):
        delays = (1000, 1500, 2000)
    else:
        delays = TARGET_DELAYS if args.fresh else sweep.DELAYS
    jobs = [(d, "LANS", "lans", "limited_lans", seed) for d in delays for seed in TRAIN_SEEDS]
    if args.fresh:
        seed_start = 33650 if args.variant in ("retained", "final") else 33550
        sweep.P["test"] = {**sweep.P["test"], "seeds": list(range(seed_start, seed_start+5)),
                           "scales": [[.9, 1.025], [1.1, .975]]}
        if args.variant in ("linear", "retained"):
            jobs.extend((d, label, kind, checkpoint, seed)
                        for d in delays for label, kind, checkpoint, seeds in sweep.CONTROLLERS
                        if label != "LANS" for seed in seeds)
            jobs.extend((d, "published", "lans", "published", seed) for d in delays for seed in TRAIN_SEEDS)
    with ProcessPoolExecutor(max_workers=4) as pool:
        records = [r for group in pool.map(run, jobs) for r in group]
    candidate = [r for r in records if r["controller"] == "LANS"]
    if args.fresh:
        if args.variant in ("nonlinear", "robust", "rounded", "final"):
            baselines = "retained_fresh_episodes.csv" if args.variant == "final" else "fresh_episodes.csv"
            with (HERE / baselines).open() as stream:
                records.extend({**row, "delay_ms": int(row["delay_ms"])}
                               for row in csv.DictReader(stream) if row["controller"] != "LANS")
        references = records
        prefix = "fresh"
    else:
        with (HERE.parent / "episodes.csv").open() as stream:
            references = [{**row, "delay_ms": int(row["delay_ms"])} for row in csv.DictReader(stream)]
        prefix = "benchmark"
    if args.variant != "linear":
        prefix = f"{args.variant}_{prefix}"
    sweep.write_csv(f"{prefix}_episodes.csv", records)
    rankings = summarize(candidate, references, delays)
    sweep.write_csv(f"{prefix}_ranking.csv", rankings)
    for row in rankings:
        if row["controller"] == "LANS":
            print(row, flush=True)


if __name__ == "__main__":
    main()
