"""Interpolate the existing delay input weights and score each fixed checkpoint set."""
import argparse

from stable_baselines3 import PPO
import torch

from distill import HERE, POLICY_SEEDS
from score import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fraction", type=float)
    args = parser.parse_args()
    if not 0. <= args.fraction <= 1.:
        parser.error("fraction must be in [0, 1]")
    torch.set_num_threads(1)
    output = HERE / "interpolated_slalom_repair" / f"{args.fraction:.3f}"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in POLICY_SEEDS:
        base = PPO.load(HERE / "repair_fixed_slalom_450_stress" / f"policy_{seed}_20.zip", device="cpu")
        anchor = PPO.load(HERE / "anchor_prefix_focus" / f"policy_{seed}_150.zip", device="cpu")
        delay_weights = base.policy.mlp_extractor.policy_net[0].weight[:, 9:11]
        anchor_weights = anchor.policy.mlp_extractor.policy_net[0].weight[:, 9:11]
        with torch.no_grad():
            delay_weights.lerp_(anchor_weights, args.fraction)
        path = output / f"policy_{seed}.zip"
        base.save(path)
        paths.append(path)
    score(paths, "calibration", HERE / "calibration" / "matched_mpc.csv", output)


if __name__ == "__main__":
    main()
