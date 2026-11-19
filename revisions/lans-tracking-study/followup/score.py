"""Rank a fixed set of three candidate policies against frozen reference episodes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
import torch

from distill import HERE, STUDY, POLICY_SEEDS, actor, make_env
from tune_head import read_means
from evaluate import cases
from batch_sim import build_cases
from calibration import ADDITIONAL_SPLITS


def score(paths, stage, mpc_path, destination, first_action_fraction=0., beta_decay=None):
    references = STUDY / "results" / stage
    names = ("lans", "ppo", "lqr", "mpc")
    rows = []
    if stage in ADDITIONAL_SPLITS:
        seeds, scales = ADDITIONAL_SPLITS[stage]
        evaluation_cases = [
            dict({k: v for k, v in c.items() if k != "seed"}, physical_seed=c["seed"])
            for c in build_cases(seeds, scales)
        ]
    else:
        evaluation_cases = list(cases(stage))
    for policy_seed, path in zip(POLICY_SEEDS, paths):
        predict = actor(PPO.load(path, device="cpu"))
        for case in evaluation_cases:
            config = dict(case)
            config["seed"] = config.pop("physical_seed")
            env = make_env(config)
            if beta_decay is not None:
                env.beta_decay = beta_decay
            obs, _ = env.reset(seed=config["seed"])
            squared_error = float(env.x[0]) ** 2
            for tick in range(1000):
                action = predict(obs)
                if tick == 0 and first_action_fraction:
                    correction = -np.sign(obs[0])
                    action += first_action_fraction * (correction - action)
                obs, _, term, trunc, info = env.step([action])
                squared_error += info["error"] ** 2
                if term or trunc:
                    break
            remaining = 999 - tick
            if remaining:
                squared_error += remaining * max(2.5, abs(info["error"])) ** 2
            rows.append(dict(
                policy_seed=policy_seed, **case, steps=tick + 1,
                completed=trunc and not term, rmse_penalized=np.sqrt(squared_error / 1001),
            ))
    destination.mkdir(exist_ok=True, parents=True)
    episode_file = destination / ("candidate_" + stage + "_episodes.csv")
    with episode_file.open("w") as stream:
        fields = csv.DictWriter(stream, fieldnames=list(rows[0]))
        fields.writeheader()
        fields.writerows(rows)
    if stage in ADDITIONAL_SPLITS:
        means = [
            read_means([episode_file]),
            *(read_means([mpc_path.with_name(f"{kind}.csv")])
              for kind in ("limited_ppo", "fast_lqr", "matched_mpc")),
        ]
    else:
        means = [
            read_means([episode_file]),
            read_means([references / f"limited_ppo_{seed}.csv" for seed in POLICY_SEEDS]),
            read_means([references / "fast_lqr_-1.csv"]),
            read_means([mpc_path or references / "matched_mpc_-1.csv"]),
        ]
    ranks = []
    for key in sorted(means[0]):
        path, delay, mass = key
        scores = [group[key] for group in means]
        record = dict(path=path, delay=delay, mass=mass)
        for name, value in zip(names, scores):
            record[f"{name}_rmse"] = value
            record[f"{name}_rank"] = 1 + sum(bool(other < value - 1e-12) for other in scores)
        ranks.append(record)
    ranks_file = destination / ("candidate_" + stage + "_ranking.csv")
    with ranks_file.open("w") as stream:
        fields = csv.DictWriter(stream, fieldnames=list(ranks[0]))
        fields.writeheader()
        fields.writerows(ranks)
    summary = dict(
        first_places=sum(row["lans_rank"] == 1 for row in ranks),
        completed=sum(row["completed"] for row in rows),
        episodes=len(rows), mean_rank=float(np.mean([r["lans_rank"] for r in ranks])),
        mean_rmse=float(np.mean([r["rmse_penalized"] for r in rows])),
    )
    (destination / ("candidate_" + stage + "_summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if beta_decay is not None:
        (destination / f"candidate_{stage}_config.json").write_text(
            json.dumps({"beta_decay_per_ms": beta_decay, "checkpoints": [str(p) for p in paths]}, indent=2) + "\n"
        )
    print(destination, stage, summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validation", "test", *ADDITIONAL_SPLITS))
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--mpc", type=Path)
    parser.add_argument("--diagnostic-first-action", type=float, default=0.,
                        help="At reset only, interpolate the legacy action toward saturated lateral correction.")
    parser.add_argument("--beta-decay", type=float,
                        help="Override the existing LANS exponential gain parameter for the candidate only.")
    args = parser.parse_args()
    if args.beta_decay is not None and (not np.isfinite(args.beta_decay) or args.beta_decay < 0):
        parser.error("--beta-decay must be finite and nonnegative")
    if not 0. <= args.diagnostic_first_action <= 1.:
        parser.error("--diagnostic-first-action must be in [0, 1]")
    if args.stage in ADDITIONAL_SPLITS and args.mpc is None:
        parser.error("--mpc is required for additional scoring splits")
    torch.set_num_threads(1)
    suffix = f"_{args.updates}" if args.updates else ""
    paths = [args.checkpoint_dir / f"policy_{seed}{suffix}.zip" for seed in POLICY_SEEDS]
    score(paths, args.stage, args.mpc, args.output_dir, args.diagnostic_first_action, args.beta_decay)
