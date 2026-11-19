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


def collect(predict, precision=False, early_refine=False):
    observations, targets, offsets, weights = [], [], [], []
    seeds = list(range(22638, 22650)) if precision else TRAIN_SEEDS
    scales = [(.88, 1.02), (1.10, .96), (1.16, 1.08)] if precision else SCALES
    if early_refine:
        seeds = list(range(22682, 22692))
        scales = [(.89, 1.02), (1.03, 1.07), (1.16, .95)]
    for case in build_cases(seeds, scales):
        if case["delay"] != "none":
            continue
        env = make_env(case)
        env.reset(seed=case["seed"])
        input_cost, rate_cost, heading_cost = (
            (.00001, .00003, .003) if early_refine else (.00005, .0001, .01)
        )
        commands = plan(env, input_cost, rate_cost, heading_cost)
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
                    offsets.append(float(env.K @ env.x))
                    weight = 1.
                    if precision:
                        if tick < 10 and case["mass"] == 1650:
                            weight = 12. if early_refine else 4.
                        elif early_refine and tick >= 40 and case["path"] == "slalom":
                            weight = 2.
                        elif tick >= 40 and case["mass"] == 1100:
                            weight = 4.
                        elif tick >= 40 and case["mass"] == 2200:
                            weight = 2.
                    weights.append(weight)
                action = steering + env.K @ env.x if reference_trajectory else predict(obs)
                obs, _, term, trunc, _ = env.step([action])
                if term or trunc:
                    break
    return (torch.tensor(np.stack(observations)), torch.tensor(targets, dtype=torch.float32),
            torch.tensor(offsets, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32))


def train(seed, precision=False, early_refine=False):
    precision = precision or early_refine
    torch.set_num_threads(1)
    directory = HERE / "direct_fifo_nominal_extended"
    path = directory / f"policy_{seed}_64.zip"
    if precision:
        path = HERE / "direct_fifo_nominal_polish" / f"policy_{seed}_48.zip"
    if early_refine:
        path = HERE / "nominal_oracle_precision" / f"policy_{seed}_4000.zip"
    model = PPO.load(path, device="cpu")
    obs, targets, offsets, weights = collect(actor(model), precision, early_refine)
    hidden = model.policy.mlp_extractor.policy_net
    head = model.policy.action_net
    rate = .000012 if early_refine else (.000035 if precision else .0003)
    optimizer = torch.optim.Adam([*hidden.parameters(), *head.parameters()], lr=rate)
    random = torch.Generator().manual_seed(seed)
    out = HERE / ("nominal_oracle_precision" if precision else "nominal_oracle_actor")
    if early_refine:
        out = HERE / "nominal_oracle_early"
    out.mkdir(exist_ok=True)
    checkpoints = (500, 1500, 4000) if precision else CHECKPOINTS
    for step in range(1, checkpoints[-1] + 1):
        index = torch.randint(len(targets), (512,), generator=random)
        predicted = head(hidden(obs[index])).squeeze(-1)
        if precision:
            predicted_command = (predicted.clamp(-1, 1) - offsets[index]).clamp(-.5, .5)
            target_command = (targets[index] - offsets[index]).clamp(-.5, .5)
            command_error = torch.nn.functional.smooth_l1_loss(
                predicted_command, target_command, beta=.008, reduction="none"
            )
            raw_error = torch.nn.functional.smooth_l1_loss(
                predicted, targets[index], beta=.02, reduction="none"
            )
            raw_weight = .035 if early_refine else .1
            loss = ((command_error + raw_weight * raw_error) * weights[index]).mean()
        else:
            loss = torch.nn.functional.smooth_l1_loss(predicted, targets[index], beta=.02)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            model.save(out / f"policy_{seed}_{step}.zip")
            print(seed, step, float(loss.detach()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    parser.add_argument("--precision", action="store_true")
    parser.add_argument("--early-refine", action="store_true")
    args = parser.parse_args()
    train(args.seed, args.precision, args.early_refine)
