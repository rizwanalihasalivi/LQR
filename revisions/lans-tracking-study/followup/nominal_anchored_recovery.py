"""Recover delayed behavior while anchoring nominal commands in the original actor."""
import argparse
import json

import numpy as np
from stable_baselines3 import PPO
import torch
from torch.func import functional_call, jacrev, vmap

from distill import HERE, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases

SEEDS = list(range(22676, 22682))
SCALES = [(.89, 1.02), (1.14, .98)]
CHECKPOINTS = (200, 800, 1600)


def collect(nominal, teacher, student):
    anchors, offsets = [], []
    dynamic = [[], []]
    targets = [[], []]
    sample_weights = [[], []]
    for case in build_cases(SEEDS, SCALES):
        env = make_env(case)
        if case["delay"] == "none":
            obs, _ = env.reset(seed=case["seed"])
            for tick in range(1000):
                action = nominal(obs)
                if tick < 70 or tick % 8 == 0:
                    anchors.append(obs)
                    offsets.append(float(env.K @ env.x))
                obs, _, term, trunc, _ = env.step([action])
                if term or trunc:
                    break
            continue

        group = int(case["delay"] == "fixed200")
        for rollout in ("teacher", "pulse", "student"):
            obs, _ = env.reset(seed=case["seed"])
            for tick in range(1000):
                label = np.clip(teacher(obs), -1., 1.)
                blind = not np.any(obs[-2:]) and env.queue.last_release > tick and env.delays[tick] <= 5
                if tick > 0 and (tick < 100 or tick % 8 == 0 or blind):
                    dynamic[group].append(obs)
                    targets[group].append(label)
                    sample_weights[group].append(3. if blind else 1.)
                if rollout == "student":
                    action = student(obs)
                elif rollout == "pulse" and tick == 0:
                    action = .5 * (label + nominal(obs))
                else:
                    action = label
                obs, _, term, trunc, _ = env.step([action])
                if term or trunc:
                    break
    return (
        torch.tensor(np.stack(anchors)),
        torch.tensor(offsets, dtype=torch.float32),
        [(torch.tensor(np.stack(x)), torch.tensor(y, dtype=torch.float32),
          torch.tensor(w, dtype=torch.float32))
         for x, y, w in zip(dynamic, targets, sample_weights, strict=True)],
    )


def train(seed):
    torch.set_num_threads(1)
    nominal_path = HERE / "nominal_oracle_precision" / f"policy_{seed}_4000.zip"
    teacher_path = HERE / "gauss_newton_joint" / f"policy_{seed}_12.zip"
    start_path = HERE / "delay_channel_recovery" / f"policy_{seed}_6000.zip"
    source = PPO.load(nominal_path, device="cpu")
    teacher = PPO.load(teacher_path, device="cpu")
    model = PPO.load(start_path, device="cpu")
    anchor, offset, data = collect(actor(source), actor(teacher), actor(model))
    reference = torch.nn.Sequential(source.policy.mlp_extractor.policy_net, source.policy.action_net)
    network = torch.nn.Sequential(model.policy.mlp_extractor.policy_net, model.policy.action_net)
    weights = dict(network.named_parameters())
    tensors = tuple(weights.values())

    def command(params, observation, base):
        raw = functional_call(network, params, (observation,)).squeeze(-1).clamp(-1., 1.)
        return (raw - base).clamp(-.5, .5)

    gradient_at = vmap(jacrev(command), in_dims=(None, 0, 0))
    with torch.no_grad():
        desired = (reference(anchor).squeeze(-1).clamp(-1., 1.) - offset).clamp(-.5, .5)
    rng = torch.Generator().manual_seed(seed)
    output = HERE / "nominal_anchored_recovery"
    output.mkdir(exist_ok=True)
    history = []
    for step in range(1, CHECKPOINTS[-1] + 1):
        sample = torch.randint(len(anchor), (128,), generator=rng)
        x, base = anchor[sample], offset[sample]
        matrices = gradient_at(weights, x, base)
        rows = torch.cat([m.reshape(len(sample), -1) for m in matrices.values()], dim=1).detach()
        values, vectors = torch.linalg.eigh(rows @ rows.T)
        keep = values > max(1e-7, float(values[-1]) * 1e-6)
        basis = (vectors[:, keep].T @ rows) / values[keep].sqrt()[:, None]

        losses = []
        for observations, target, confidence in data:
            subset = torch.randint(len(target), (512,), generator=rng)
            error = torch.nn.functional.smooth_l1_loss(
                network(observations[subset]).squeeze(-1), target[subset],
                beta=.012, reduction="none",
            )
            losses.append((error * confidence[subset]).mean())
        loss = torch.stack(losses).mean()
        gradient = torch.cat([g.flatten() for g in torch.autograd.grad(loss, tensors)])
        direction = gradient - basis.T @ (basis @ gradient)
        direction *= .0016 / direction.norm().clamp_min(1e-12)
        with torch.no_grad():
            drift = command(weights, x, base) - desired[sample]
            correction = basis.T @ ((vectors[:, keep].T @ drift) / values[keep].sqrt())
            direction += min(.3, .001 / max(float(correction.norm()), 1e-12)) * correction
            for parameter, update in zip(tensors, direction.split([p.numel() for p in tensors]), strict=True):
                parameter.sub_(update.reshape_as(parameter))
        if step in CHECKPOINTS:
            model.save(output / f"policy_{seed}_{step}.zip")
            record = {"step": step, "losses": [float(value.detach()) for value in losses],
                      "nominal_anchor_mse": float(drift.square().mean())}
            history.append(record)
            print(seed, record, flush=True)
    (output / f"training_{seed}.json").write_text(json.dumps({
        "source": str(start_path.relative_to(HERE)),
        "nominal_anchor": str(nominal_path.relative_to(HERE)),
        "delayed_teacher": str(teacher_path.relative_to(HERE)),
        "training_seeds": SEEDS,
        "scales": SCALES,
        "history": history,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    train(parser.parse_args().seed)
