"""Train the unchanged actor on progressively longer transport delays."""
import argparse
import json

import numpy as np
from stable_baselines3 import PPO
import torch

from distill import HERE, POLICY_SEEDS
from batch_sim import build_cases, prepare
from core import KR, XS, beta
from features import T

SEEDS = [22588, 22589, 22590, 22591]
SCALES = [(.87, 1.01), (1.13, .98)]
HORIZON = 200
LEVELS = (.08, .2, .45, .7, 1.)
UPDATES_PER_LEVEL = 24
CHECKPOINTS = (48, 96, 120)


def inputs(delays):
    steps = np.arange(HORIZON)
    releases = np.maximum.accumulate(steps[None, :] + np.ceil(delays[:, :HORIZON] / 10).astype(int), axis=1)
    applied = np.stack([np.searchsorted(row, steps, side="right") for row in releases]).T
    window = int((steps[:, None] + 1 - applied).max()) + 1
    index = applied - np.maximum(steps + 2 - window, 0)[:, None]
    observed = np.zeros((HORIZON, len(delays), 2))
    for tick in range(1, HORIZON):
        observed[tick, :, 0] = delays[:, tick - 1] / 200
        observed[tick, :, 1] = delays[:, max(0, tick - 10):tick].mean(axis=1) / 200
    decay = beta(delays[:, :HORIZON].T)
    return torch.tensor(index), window, torch.tensor(observed), torch.tensor(decay)


def train(seed):
    torch.set_num_threads(1)
    source = HERE / "direct_fifo_nominal_polish" / f"policy_{seed}_48.zip"
    model = PPO.load(source, device="cpu")
    hidden = model.policy.mlp_extractor.policy_net
    head = model.policy.action_net
    parameters = [*hidden.parameters(), *head.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=.00013)
    cases = build_cases(SEEDS, SCALES)
    data = prepare(cases)
    transition, steering, initial = (torch.tensor(data[k]) for k in ("A", "B", "initial"))
    references = torch.tensor(data["ref"])
    disturbances = (
        data["G"][:, :, None] * data["ref"][:, None, 2, :HORIZON]
        + data["H"][:, :, None] * data["ref"][:, None, 3, :HORIZON]
    ) / 25
    disturbances = torch.tensor(disturbances.transpose(2, 0, 1))
    horizons = torch.arange(1, 5, dtype=torch.float64) * .01
    speed = torch.ones((len(cases), 1), dtype=torch.float64)
    scale, gain, whitening = map(torch.tensor, (XS, KR, T))
    episode = torch.arange(len(cases))
    weights = torch.tensor([[case["delay"] == mode for case in cases]
                            for mode in ("none", "mixed", "fixed200")], dtype=torch.float64)
    weights /= weights.sum(dim=1, keepdim=True)

    def rollout(schedule):
        index, window, observed, decay = schedule
        state = initial
        error = torch.zeros(len(cases), dtype=torch.float64)
        commands = [torch.zeros_like(error)]
        for tick in range(HORIZON):
            offsets = state[:, 0, None] + 25 * state[:, 2, None] * horizons
            preview = (references[:, 0, tick, None] + offsets
                       + references[:, 1, tick, None] * horizons
                       - references[:, 0, tick + 1:tick + 5]).float().double()
            channels = (preview - offsets) / (10 * horizons**2)
            conditioned = channels.float().double() @ whitening
            observation = torch.cat((state / scale, speed, conditioned, observed[tick]), dim=-1).float()
            residual = head(hidden(observation)).squeeze(-1).double().clamp(-1, 1)
            command = (decay[tick] * residual - state @ gain).clamp(-.5, .5)
            commands.append(command)
            applied = torch.stack(commands[-window:])[index[tick], episode]
            state = (torch.einsum("nij,nj->ni", transition, state) + steering * applied[:, None]
                     + disturbances[tick])
            error = error + state[:, 0].square()
        return weights @ (error / HORIZON)

    output = HERE / "delay_curriculum"
    output.mkdir(exist_ok=True)
    records = []
    for step in range(1, CHECKPOINTS[-1] + 1):
        if (step - 1) % UPDATES_PER_LEVEL == 0:
            factor = LEVELS[(step - 1) // UPDATES_PER_LEVEL]
            schedule = inputs(data["delays"] * factor)
        error = rollout(schedule)
        loss = torch.log1p(error / torch.tensor([.00002, .00008, .0003])).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        if step in CHECKPOINTS:
            model.save(output / f"policy_{seed}_{step}.zip")
            record = {"step": step, "delay_fraction": factor, "training_mse": error.detach().tolist()}
            records.append(record)
            print(seed, record, flush=True)
    (output / f"training_{seed}.json").write_text(json.dumps({
        "source": str(source.relative_to(HERE)),
        "seeds": SEEDS,
        "scales": SCALES,
        "horizon": HORIZON,
        "snapshots": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    train(parser.parse_args().seed)
