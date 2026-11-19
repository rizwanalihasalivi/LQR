"""Initialize one existing 64x64 actor from compact observation-conditioned fits."""
import argparse
import json

import numpy as np
from stable_baselines3 import PPO
import torch

from distill import HERE, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases

SEEDS = [22584, 22585, 22586, 22587]
SCALES = [(.90, 1.05), (1.14, .94)]
WIDTH = 30
THRESHOLD = .001
CHECKPOINTS = (1500, 4500)


def collect(prior, nominal):
    observations = ([], [])
    targets = ([], [])
    for case in build_cases(SEEDS, SCALES):
        for teacher_rollout in (False, True):
            env = make_env(case)
            obs, _ = env.reset(seed=case["seed"])
            nominal_case = case["delay"] == "none"
            for tick in range(1000):
                recent_delay = obs[9] + obs[10]
                group = int(recent_delay >= THRESHOLD)
                nominal_action = np.clip(nominal(obs), -1., 1.)
                delayed_action = np.clip(prior(obs), -1., 1.)
                if tick < 100 or tick % 8 == 0:
                    observations[group].append(obs)
                    label = nominal_action if nominal_case or tick == 0 else delayed_action
                    targets[group].append(label)
                if teacher_rollout and (nominal_case or tick == 0):
                    action = nominal_action
                else:
                    action = delayed_action
                obs, _, terminated, truncated, _ = env.step([action])
                if terminated or truncated:
                    break
    return [(torch.tensor(np.stack(x)), torch.tensor(y, dtype=torch.float32))
            for x, y in zip(observations, targets, strict=True)]


def compact(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(11, WIDTH), torch.nn.ReLU(),
        torch.nn.Linear(WIDTH, WIDTH), torch.nn.ReLU(),
        torch.nn.Linear(WIDTH, 1),
    )


def export_actor(model, fits, path):
    hidden = model.policy.mlp_extractor.policy_net
    first, second, output = hidden[0], hidden[2], model.policy.action_net
    assert first.weight.shape == (64, 11)
    assert second.weight.shape == (64, 64)
    with torch.no_grad():
        for layer in (first, second, output):
            layer.weight.zero_()
            layer.bias.zero_()
        # These two activations are ordinary units in the first hidden layer.
        # Their inputs are the existing last and recent observed delays.
        first.weight[60, 9:11] = 1 / THRESHOLD
        first.weight[61, 9:11] = -1 / THRESHOLD
        first.bias[61] = 1.
        for group, net in enumerate(fits):
            start = group * WIDTH
            units = slice(start, start + WIDTH)
            delay_unit = 60 + group
            bias_unit = 60 + group
            first.weight[units] = net[0].weight
            first.bias[units] = net[0].bias
            second.weight[units, units] = net[2].weight
            second.bias[units] = net[2].bias
            second.weight[units, delay_unit] = -32.
            second.weight[bias_unit, delay_unit] = -2.
            second.bias[bias_unit] = 1.
            output.weight[:, units] = net[4].weight
            output.weight[0, bias_unit] = net[4].bias[0]
        model.save(path)


def train(seed):
    torch.set_num_threads(1)
    source = HERE / "guarded_round2" / "blend_.1" / f"policy_{seed}.zip"
    expert = HERE / "direct_fifo_nominal_polish" / f"policy_{seed}_48.zip"
    model = PPO.load(source, device="cpu")
    data = collect(actor(model), actor(PPO.load(expert, device="cpu")))
    fits = (compact(seed), compact(seed + 100))
    optimizers = [torch.optim.Adam(fit.parameters(), lr=.001) for fit in fits]
    generator = torch.Generator().manual_seed(seed)
    directory = HERE / "actor_context_init"
    directory.mkdir(exist_ok=True)
    records = []
    for step in range(1, CHECKPOINTS[-1] + 1):
        losses = []
        for fit, opt, (obs, target) in zip(fits, optimizers, data, strict=True):
            index = torch.randint(len(obs), (768,), generator=generator)
            prediction = fit(obs[index]).squeeze(-1)
            loss = torch.nn.functional.smooth_l1_loss(prediction, target[index], beta=.012)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if step in CHECKPOINTS:
            export_actor(model, fits, directory / f"policy_{seed}_{step}.zip")
            records.append({"step": step, "losses": losses})
            print(seed, step, losses, flush=True)
    (directory / f"training_{seed}.json").write_text(json.dumps({
        "source": str(source.relative_to(HERE)),
        "nominal_teacher": str(expert.relative_to(HERE)),
        "training_seeds": SEEDS,
        "scales": SCALES,
        "fit_width": WIDTH,
        "delay_threshold": THRESHOLD,
        "samples_by_context": [len(x) for x, _ in data],
        "snapshots": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    train(parser.parse_args().seed)
