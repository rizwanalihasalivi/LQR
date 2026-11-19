"""Transfer nominal-only steering targets while anchoring delayed outputs."""
import argparse

import numpy as np
from stable_baselines3 import PPO
import torch

from distill import HERE, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases

SEEDS = [22543, 22544, 22545]
SCALES = [(.92, 1.08), (1.08, .94)]
CHECKPOINTS = (100, 400, 1000)


def main(seed, bounded_teacher=False):
    torch.set_num_threads(1)
    source = PPO.load(HERE / "guarded_round2" / "blend_.1" / f"policy_{seed}.zip", device="cpu")
    if bounded_teacher:
        teacher_path = HERE / "nominal_oracle_actor" / f"policy_{seed}_2400.zip"
        training_seeds = [22550, 22551, 22552]
        scales = [(.90, .99), (1.11, 1.07)]
        fraction = .025
    else:
        teacher_path = HERE / "direct_fifo_nominal_extended" / f"policy_{seed}_64.zip"
        training_seeds = SEEDS
        scales = SCALES
        fraction = .25
    expert = PPO.load(teacher_path, device="cpu")
    prior, nominal_expert = actor(source), actor(expert)
    observations, labels = [], []
    for case in build_cases(training_seeds, scales):
        env = make_env(case)
        obs, _ = env.reset(seed=case["seed"])
        nominal = case["delay"] == "none"
        for tick in range(1000):
            action = prior(obs)
            if tick < 150 or tick % 8 == 0:
                observations.append(obs)
                labels.append(action + .25 * (nominal_expert(obs) - action) if nominal else action)
            obs, _, terminated, truncated, _ = env.step([action])
            if terminated or truncated:
                break
    observations = torch.tensor(np.stack(observations))
    labels = torch.tensor(labels, dtype=torch.float32)
    hidden = source.policy.mlp_extractor.policy_net
    head = source.policy.action_net
    optimizer = torch.optim.Adam([*hidden.parameters(), *head.parameters()], lr=.000025)
    generator = torch.Generator().manual_seed(seed)
    output = HERE / ("bounded_nominal_transfer" if bounded_teacher else "nominal_transfer")
    output.mkdir(exist_ok=True)
    for step in range(1, CHECKPOINTS[-1] + 1):
        idx = torch.randint(len(labels), (512,), generator=generator)
        predicted = head(hidden(observations[idx])).squeeze(-1)
        loss = (predicted - labels[idx]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in CHECKPOINTS:
            source.save(output / f"policy_{seed}_{step}.zip")
            print(seed, step, float(loss.detach()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    parser.add_argument("--bounded-teacher", action="store_true")
    args = parser.parse_args()
    main(args.seed, args.bounded_teacher)
