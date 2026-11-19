"""Interpolate only the deployed actor weights between fixed checkpoints."""
import argparse
from pathlib import Path

from stable_baselines3 import PPO
import torch

from distill import HERE, POLICY_SEEDS


def interpolate(fraction, output, second_pass=False, nominal_focus=False, nominal_transfer=False):
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be within [0, 1]")
    output.mkdir(parents=True, exist_ok=False)
    for seed in POLICY_SEEDS:
        if nominal_transfer:
            source_path = HERE / "guarded_round2" / "blend_.1" / f"policy_{seed}.zip"
            teacher_path = HERE / "nominal_transfer" / f"policy_{seed}_400.zip"
        elif nominal_focus:
            source_path = HERE / "guarded_round2" / "blend_.1" / f"policy_{seed}.zip"
            teacher_path = HERE / "nominal_focused" / f"policy_{seed}_1000.zip"
        elif second_pass:
            source_path = HERE / "guarded_distilled" / f"policy_{seed}_600.zip"
            teacher_path = HERE / "guarded_round2" / f"policy_{seed}_600.zip"
        else:
            source_path = HERE / "initial_trial" / f"policy_{seed}.zip"
            teacher_path = HERE / "constrained_distilled" / f"policy_{seed}_400.zip"
        source = PPO.load(source_path, device="cpu")
        teacher = PPO.load(teacher_path, device="cpu")
        with torch.no_grad():
            source_actor = source.policy.mlp_extractor.policy_net
            target_actor = teacher.policy.mlp_extractor.policy_net
            actor_pairs = (
                (source.policy.action_net, teacher.policy.action_net),
                (source_actor, target_actor),
            )
            for current, target in actor_pairs:
                for p, q in zip(current.parameters(), target.parameters(), strict=True):
                    p.lerp_(q, fraction)
        source.save(output / f"policy_{seed}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fraction", type=float)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--round2", action="store_true")
    mode.add_argument("--nominal-focus", action="store_true")
    mode.add_argument("--nominal-transfer", action="store_true")
    args = parser.parse_args()
    interpolate(args.fraction, args.output, args.round2, args.nominal_focus, args.nominal_transfer)
