"""Optimize early nominal error through the simulator's discrete dynamics."""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
import torch

from distill import HERE, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases, prepare
from core import KR, XS
from features import T

HORIZON = 60
TRAINING_SEEDS = [22518, 22519, 22520, 22521]
SCALES = [(.92, 1.03), (1.11, .96)]


def main(seed):
    torch.set_num_threads(1)
    source = HERE / "guarded_round2" / "blend_.1" / f"policy_{seed}.zip"
    model = PPO.load(source, device="cpu")
    predict = actor(model)
    cases = build_cases(TRAINING_SEEDS, SCALES)
    nominal = [case for case in cases if case["delay"] == "none"]
    data = prepare(nominal)
    transition, steer, initial = (torch.tensor(data[k]) for k in ("A", "B", "initial"))
    refs = torch.tensor(data["ref"])
    disturbances = (
        data["G"][:, :, None] * data["ref"][:, None, 2, :HORIZON]
        + data["H"][:, :, None] * data["ref"][:, None, 3, :HORIZON]
    ) / 25
    disturbances = torch.tensor(disturbances.transpose(2, 0, 1))
    scale = torch.tensor(XS)
    gain = torch.tensor(KR)
    whitening = torch.tensor(T)
    horizons = torch.arange(1, 5, dtype=torch.float64) * .01
    speed = torch.ones((len(nominal), 1), dtype=torch.float64)
    delay = torch.zeros((len(nominal), 2), dtype=torch.float64)
    hidden = model.policy.mlp_extractor.policy_net
    head = model.policy.action_net

    observations = []
    for case in cases:
        if case["delay"] == "none":
            continue
        env = make_env(case)
        obs, _ = env.reset(seed=case["seed"])
        for tick in range(500):
            if tick < 75 or tick % 5 == 0:
                observations.append(obs)
            obs, _, term, trunc, _ = env.step([predict(obs)])
            if term or trunc:
                break
    with torch.no_grad():
        anchors = hidden(torch.tensor(np.stack(observations)))
        anchor_actions = head(anchors).squeeze(-1).clone()

    def rollout():
        state = initial
        loss = 0.
        for tick in range(HORIZON):
            preview = (
                refs[:, 0, tick, None] + state[:, 0, None]
                + (refs[:, 1, tick, None] + 25 * state[:, 2, None]) * horizons
                - refs[:, 0, tick + 1:tick + 5]
            ).float().double()
            channels = (
                preview - state[:, 0, None] - 25 * state[:, 2, None] * horizons
            ) / (10 * horizons**2)
            conditioned = channels.float().double() @ whitening
            obs = torch.cat((state / scale, speed, conditioned, delay), dim=-1).float()
            command = (head(hidden(obs)).squeeze(-1).double().clamp(-1, 1) - state @ gain)
            command = command.clamp(-.5, .5)
            state = torch.einsum("nij,nj->ni", transition, state) + steer * command[:, None]
            state = state + disturbances[tick]
            loss = loss + state[:, 0].square().mean()
        return loss / HORIZON

    for parameter in hidden.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(head.parameters(), lr=.000025)
    nominal_scale = float(rollout().detach())
    out = HERE / "direct_transient"
    out.mkdir(exist_ok=True)
    checkpoints = (25, 75, 150)
    for step in range(1, checkpoints[-1] + 1):
        transient = rollout() / nominal_scale
        drift = (head(anchors).squeeze(-1) - anchor_actions).square().mean()
        loss = transient + 25000 * drift
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            model.save(out / f"policy_{seed}_{step}.zip")
            print(seed, step, float(transient.detach()), float(drift.detach()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    main(parser.parse_args().seed)
