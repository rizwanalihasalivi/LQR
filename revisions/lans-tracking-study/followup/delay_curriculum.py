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


def inputs(delays, decay_per_ms=None):
    steps = np.arange(HORIZON)
    releases = np.maximum.accumulate(steps[None, :] + np.ceil(delays[:, :HORIZON] / 10).astype(int), axis=1)
    applied = np.stack([np.searchsorted(row, steps, side="right") for row in releases]).T
    window = int((steps[:, None] + 1 - applied).max()) + 1
    index = applied - np.maximum(steps + 2 - window, 0)[:, None]
    observed = np.zeros((HORIZON, len(delays), 2))
    for tick in range(1, HORIZON):
        observed[tick, :, 0] = delays[:, tick - 1] / 200
        observed[tick, :, 1] = delays[:, max(0, tick - 10):tick].mean(axis=1) / 200
    current_delay = delays[:, :HORIZON].T
    decay = (
        beta(current_delay) if decay_per_ms is None
        else np.exp(-decay_per_ms * np.maximum(0., current_delay - 5.))
    )
    return torch.tensor(index), window, torch.tensor(observed), torch.tensor(decay)


def train(seed, gain_protected=False, intermediate_gain=False):
    gain_protected = gain_protected or intermediate_gain
    torch.set_num_threads(1)
    source = HERE / "direct_fifo_nominal_polish" / f"policy_{seed}_48.zip"
    if gain_protected:
        source = HERE / "delay_channel_recovery" / f"policy_{seed}_6000.zip"
    model = PPO.load(source, device="cpu")
    hidden = model.policy.mlp_extractor.policy_net
    head = model.policy.action_net
    parameters = [*hidden.parameters(), *head.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=.00007 if gain_protected else .00013)
    seeds = list(range(22692, 22697)) if gain_protected else SEEDS
    scales = [(.89, 1.05), (1.12, .97)] if gain_protected else SCALES
    if intermediate_gain:
        seeds = list(range(22697, 22702))
        scales = [(.90, .96), (1.14, 1.07)]
    cases = build_cases(seeds, scales)
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
    if gain_protected:
        output = HERE / "delay_curriculum_protected_gain"
    if intermediate_gain:
        output = HERE / "delay_curriculum_intermediate_gain"
    output.mkdir(exist_ok=True)
    records = []
    levels = (.25, .5, .75, 1.) if gain_protected else LEVELS
    stage_length = 32 if gain_protected else UPDATES_PER_LEVEL
    checkpoints = (64, 96, 128) if gain_protected else CHECKPOINTS
    decay_per_ms = .008 if intermediate_gain else (.025 if gain_protected else None)
    with torch.no_grad():
        nominal_reference = rollout(inputs(data["delays"], decay_per_ms))[0].clamp_min(1e-10)
    for step in range(1, checkpoints[-1] + 1):
        if (step - 1) % stage_length == 0:
            stage = (step - 1) // stage_length
            factor = levels[stage]
            training_gain = (.025, .018, .012, .008)[stage] if intermediate_gain else decay_per_ms
            schedule = inputs(data["delays"] * factor, training_gain)
        error = rollout(schedule)
        loss = torch.log1p(error / torch.tensor([.00002, .00008, .0003])).mean()
        if gain_protected:
            delayed_loss = torch.log1p(error[1:] / torch.tensor([.00008, .0003])).mean()
            nominal_ratio = error[0] / nominal_reference
            loss = delayed_loss + .6 * nominal_ratio + 15 * (nominal_ratio - 1.01).clamp_min(0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        if step in checkpoints:
            model.save(output / f"policy_{seed}_{step}.zip")
            record = {"step": step, "delay_fraction": factor, "training_gain": training_gain, "training_mse": error.detach().tolist()}
            records.append(record)
            print(seed, record, flush=True)
    (output / f"training_{seed}.json").write_text(json.dumps({
        "source": str(source.relative_to(HERE)),
        "seeds": seeds,
        "scales": scales,
        "beta_decay_per_ms": decay_per_ms,
        "horizon": HORIZON,
        "snapshots": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    parser.add_argument("--gain-protected", action="store_true")
    parser.add_argument("--intermediate-gain", action="store_true")
    args = parser.parse_args()
    train(args.seed, args.gain_protected, args.intermediate_gain)
