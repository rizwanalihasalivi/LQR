"""Fit the original actor to bounded nominal steering on offline trajectories."""
import argparse

import numpy as np
from stable_baselines3 import PPO
import torch

from distill import HERE, POLICY_SEEDS, actor, make_env, offline_expert
from batch_sim import build_cases
from constrained_expert import plan

TRAIN_SEEDS = [22546, 22547, 22548, 22549]
SCALES = [(.91, 1.04), (1.12, .94)]
CHECKPOINTS = (300, 1000, 2400)


def collect(predict):
    observations, targets = [], []
    for case in build_cases(TRAIN_SEEDS, SCALES):
        if case["delay"] != "none":
            continue
        env = make_env(case)
        env.reset(seed=case["seed"])
        commands = plan(env, .00005, .0001)
        feedback, _ = offline_expert(env)
        optimal_states = []
        for tick, steering in enumerate(commands):
            optimal_states.append(env.x.copy())
            env.step([steering + env.K @ env.x])
        for reference_trajectory in (True, False):
            obs, _ = env.reset(seed=case["seed"])
            for tick, steering in enumerate(commands):
                recovery = np.clip(steering - feedback @ (env.x - optimal_states[tick]), -.5, .5)
                label = np.clip(recovery + env.K @ env.x, -1., 1.)
                if tick < 60 or tick % 6 == 0:
                    observations.append(obs)
                    targets.append(label)
                action = steering + env.K @ env.x if reference_trajectory else predict(obs)
                obs, _, term, trunc, _ = env.step([action])
                if term or trunc:
                    break
    return torch.tensor(np.stack(observations)), torch.tensor(targets, dtype=torch.float32)


def train(seed):
    torch.set_num_threads(1)
    directory = HERE / "direct_fifo_nominal_extended"
    model = PPO.load(directory / f"policy_{seed}_64.zip", device="cpu")
    obs, targets = collect(actor(model))
    hidden = model.policy.mlp_extractor.policy_net
    head = model.policy.action_net
    optimizer = torch.optim.Adam([*hidden.parameters(), *head.parameters()], lr=.0003)
    random = torch.Generator().manual_seed(seed)
    out = HERE / "nominal_oracle_actor"
    out.mkdir(exist_ok=True)
    for step in range(1, CHECKPOINTS[-1] + 1):
        index = torch.randint(len(targets), (512,), generator=random)
        predicted = head(hidden(obs[index])).squeeze(-1)
        loss = torch.nn.functional.smooth_l1_loss(predicted, targets[index], beta=.02)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in CHECKPOINTS:
            model.save(out / f"policy_{seed}_{step}.zip")
            print(seed, step, float(loss.detach()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    train(parser.parse_args().seed)
