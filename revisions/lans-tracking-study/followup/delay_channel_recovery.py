"""Fit only the existing actor's two delay input columns."""
import argparse
import json

import numpy as np
from stable_baselines3 import PPO
import torch

from distill import HERE, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases

SEEDS = list(range(22668, 22676))
SCALES = [(.90, 1.04), (1.13, .95)]
CHECKPOINTS = (600, 2000, 6000)


def collect(nominal_actor, delayed_teacher):
    observations = [[], []]
    targets = [[], []]
    for case in build_cases(SEEDS, SCALES):
        if case["delay"] == "none":
            continue
        group = int(case["delay"] == "fixed200")
        env = make_env(case)
        for initial_strength in (0., .4, 1.):
            obs, _ = env.reset(seed=case["seed"])
            for tick in range(1000):
                target = np.clip(delayed_teacher(obs), -1., 1.)
                if max(obs[9], obs[10]) > 0. and (tick < 120 or tick % 8 == 0):
                    observations[group].append(obs)
                    targets[group].append(target)
                action = target if tick else target + initial_strength * (nominal_actor(obs) - target)
                obs, _, terminated, truncated, _ = env.step([action])
                if terminated or truncated:
                    break
    return [
        (torch.tensor(np.stack(obs)), torch.tensor(labels, dtype=torch.float32))
        for obs, labels in zip(observations, targets, strict=True)
    ]


def train(seed):
    torch.set_num_threads(1)
    nominal_path = HERE / "nominal_oracle_precision" / f"policy_{seed}_4000.zip"
    teacher_path = HERE / "gauss_newton_joint" / f"policy_{seed}_12.zip"
    model = PPO.load(nominal_path, device="cpu")
    teacher = PPO.load(teacher_path, device="cpu")
    samples = collect(actor(model), actor(teacher))
    hidden = model.policy.mlp_extractor.policy_net
    first, second = hidden[0], hidden[2]
    head = model.policy.action_net
    fixed_input = first.weight[:, :-2].detach().clone()
    delay_columns = torch.nn.Parameter(first.weight[:, -2:].detach().clone())
    first_bias = first.bias.detach().clone()
    for layer in (second, head):
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam([delay_columns], lr=.007)
    generator = torch.Generator().manual_seed(seed)
    out = HERE / "delay_channel_recovery"
    out.mkdir(exist_ok=True)
    history = []

    def action(obs):
        values = obs[:, :-2] @ fixed_input.T + obs[:, -2:] @ delay_columns.T + first_bias
        return head(second(values.relu()).relu()).squeeze(-1)

    for step in range(1, CHECKPOINTS[-1] + 1):
        losses = []
        for obs, labels in samples:
            index = torch.randint(len(labels), (768,), generator=generator)
            loss = torch.nn.functional.smooth_l1_loss(action(obs[index]), labels[index], beta=.012)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in CHECKPOINTS:
            with torch.no_grad():
                first.weight[:, -2:].copy_(delay_columns)
                assert torch.equal(first.weight[:, :-2], fixed_input)
            model.save(out / f"policy_{seed}_{step}.zip")
            record = {"step": step, "training_loss": [float(item.detach()) for item in losses]}
            history.append(record)
            print(seed, record, flush=True)
    (out / f"training_{seed}.json").write_text(json.dumps({
        "source": str(nominal_path.relative_to(HERE)),
        "teacher": str(teacher_path.relative_to(HERE)),
        "training_seeds": SEEDS,
        "scales": SCALES,
        "history": history,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    train(parser.parse_args().seed)
