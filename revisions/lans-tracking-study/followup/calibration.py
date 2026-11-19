"""Generate frozen comparison results on a broader calibration split."""
import csv
from pathlib import Path
import sys
import numpy as np
from stable_baselines3 import PPO
import torch

from distill import STUDY, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases
from dynamic_env import design
from matched_mpc import MatchedMPC

SEEDS = [22800, 22801, 22802, 22803, 22804, 22805]
SCALES = [(.85, .91), (1.16, 1.09)]
HELDOUT_SEEDS = [22950, 22951, 22952, 22953, 22954]
HELDOUT_SCALES = [(.90, 1.02), (1.14, .94)]
CONFIRMATION_SEEDS = [23080, 23081, 23082, 23083, 23084]
CONFIRMATION_SCALES = [(.94, 1.07), (1.10, .97)]
ADDITIONAL_SPLITS = {
    "calibration": (SEEDS, SCALES),
    "heldout": (HELDOUT_SEEDS, HELDOUT_SCALES),
    "confirmation": (CONFIRMATION_SEEDS, CONFIRMATION_SCALES),
}


def generate(kind, path, seeds=SEEDS, scales=SCALES):
    torch.set_num_threads(1)
    policy_seeds = POLICY_SEEDS if kind == "limited_ppo" else (-1,)
    rows = []
    for seed in policy_seeds:
        if kind == "limited_ppo":
            policy = PPO.load(STUDY / "models" / f"limited_ppo_{seed}" / "policy.zip", device="cpu")
            predict = actor(policy)
        for case in build_cases(seeds, scales):
            env = make_env(case)
            env.variant = {"limited_ppo": "ppo", "fast_lqr": "lqr", "matched_mpc": "mpc"}[kind]
            env.residual_scale = .5
            if kind == "fast_lqr":
                env.K = design()
            mpc = MatchedMPC() if kind == "matched_mpc" else None
            obs, _ = env.reset(seed=case["seed"])
            sse = float(env.x[0])**2
            for tick in range(1000):
                action = predict(obs) if kind == "limited_ppo" else 0.
                if mpc is not None:
                    action = mpc.action(env) / .5
                obs, _, term, trunc, info = env.step([action])
                sse += info["error"]**2
                if term or trunc:
                    break
            sse += (999 - tick) * max(2.5, abs(info["error"]))**2
            rows.append(dict(
                policy_seed=seed, **case, completed=trunc and not term,
                rmse_penalized=float(np.sqrt(sse / 1001)),
            ))
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(kind, len(rows), flush=True)


if __name__ == "__main__":
    if "--confirmation" in sys.argv:
        generate(sys.argv[1], Path(sys.argv[2]), *ADDITIONAL_SPLITS["confirmation"])
    elif "--heldout" in sys.argv:
        generate(sys.argv[1], Path(sys.argv[2]), HELDOUT_SEEDS, HELDOUT_SCALES)
    else:
        generate(sys.argv[1], Path(sys.argv[2]))
