"""Teach the first few nominal commands while anchoring the later learned behavior."""
import argparse
import json

import numpy as np
from stable_baselines3 import PPO
import torch
from torch.func import functional_call, jacrev, vmap

from distill import HERE, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases
from receding_nominal_targets import SteeringTeacher

SEEDS = list(range(22738, 22750))
SCALES = [(.89, 1.03), (1.06, .94), (1.15, 1.07)]
CHECKPOINTS = (200, 800, 1600)


def collect(predict, wide_resets=False):
    prefix, anchors = [], []
    for case in build_cases(SEEDS, SCALES):
        if case["delay"] != "none":
            continue
        env = make_env(case)
        teacher = SteeringTeacher(env) if case["mass"] == 1650 else None
        for expert_path in (False, True) if teacher else (False,):
            obs, _ = env.reset(seed=case["seed"])
            for tick in range(1000):
                offset = float(env.K @ env.x)
                action = predict(obs)
                if teacher and tick < 8:
                    target = teacher.action(env, tick)
                    prefix.append((obs, offset, target))
                    if expert_path:
                        action = target + offset
                elif tick < 65 or tick % 10 == 0:
                    anchors.append((obs, offset, np.clip(np.clip(action, -1, 1) - offset, -.5, .5)))
                obs, _, terminated, truncated, _ = env.step([action])
                if terminated or truncated:
                    break
    if wide_resets:
        # Add short trajectories spanning substantially more independent reset states.
        for group in range(3):
            seeds = list(range(23200 + 32 * group, 23232 + 32 * group))
            for case in build_cases(seeds, [SCALES[group]]):
                if case["delay"] != "none" or case["mass"] != 1650:
                    continue
                env = make_env(case)
                teacher = SteeringTeacher(env)
                for expert_path in (False, True):
                    obs, _ = env.reset(seed=case["seed"])
                    for tick in range(8):
                        offset = float(env.K @ env.x)
                        target = teacher.action(env, tick)
                        prefix.append((obs, offset, target))
                        action = target + offset if expert_path else predict(obs)
                        obs, _, terminated, _, _ = env.step([action])
                        if terminated:
                            break
    def tensors(records):
        obs, offset, value = zip(*records)
        return tuple(torch.tensor(np.asarray(a), dtype=torch.float32) for a in (obs, offset, value))
    return tensors(prefix), tensors(anchors)


def train(seed, wide_resets=False, polish=False):
    torch.set_num_threads(1)
    path = HERE / "receding_nominal_targets" / f"policy_{seed}_5000.zip"
    if polish:
        path = HERE / "receding_wide_resets" / f"policy_{seed}_1600.zip"
        wide_resets = True
    model = PPO.load(path, device="cpu")
    prefix, anchors = collect(actor(model), wide_resets)
    network = torch.nn.Sequential(model.policy.mlp_extractor.policy_net, model.policy.action_net)
    named = dict(network.named_parameters())
    parameters = tuple(named.values())
    sizes = [p.numel() for p in parameters]

    def command(weights, obs, offset):
        raw = functional_call(network, weights, (obs,)).squeeze(-1).clamp(-1, 1)
        return (raw - offset).clamp(-.5, .5)

    jacobian = vmap(jacrev(command), in_dims=(None, 0, 0))
    generator = torch.Generator().manual_seed(seed + 1000 if polish else seed)
    out = HERE / ("receding_wide_resets" if wide_resets else "receding_prefix_refine")
    if polish:
        out = HERE / "receding_wide_polish"
    out.mkdir(exist_ok=True)
    records = []
    checkpoints = (400, 1200, 3200) if polish else CHECKPOINTS
    for step in range(1, checkpoints[-1] + 1):
        chosen = torch.randint(len(anchors[0]), (128,), generator=generator)
        observations, offset, desired = (x[chosen] for x in anchors)
        matrices = jacobian(named, observations, offset)
        rows = torch.cat([value.reshape(len(chosen), -1) for value in matrices.values()], dim=1).detach()
        eig, vec = torch.linalg.eigh(rows @ rows.T)
        keep = eig > max(1e-7, float(eig[-1]) * 1e-6)
        basis = (vec[:, keep].T @ rows) / eig[keep].sqrt()[:, None]
        sample = torch.randint(len(prefix[0]), (512,), generator=generator)
        observation, base, target = (x[sample] for x in prefix)
        raw = network(observation).squeeze(-1)
        steering = (raw.clamp(-1, 1) - base).clamp(-.5, .5)
        command_error = torch.nn.functional.smooth_l1_loss(steering, target, beta=.004)
        # Keep a derivative on incorrectly saturated outputs.
        raw_error = torch.nn.functional.smooth_l1_loss(raw, (target + base).clamp(-1, 1), beta=.015)
        objective = command_error + .015 * raw_error
        gradient = torch.cat([g.flatten() for g in torch.autograd.grad(objective, parameters)])
        direction = gradient - basis.T @ (basis @ gradient)
        direction *= (.0001 if polish else .0002) / direction.norm().clamp_min(1e-12)
        with torch.no_grad():
            deviation = command(named, observations, offset) - desired
            correction = basis.T @ ((vec[:, keep].T @ deviation) / eig[keep].sqrt())
            direction += min(.3, .0003 / max(float(correction.norm()), 1e-12)) * correction
            for parameter, delta in zip(parameters, direction.split(sizes), strict=True):
                parameter.sub_(delta.reshape_as(parameter))
        if step in checkpoints:
            model.save(out / f"policy_{seed}_{step}.zip")
            record = {"step": step, "prefix_loss": float(command_error.detach()),
                      "anchor_error": float(deviation.abs().mean())}
            records.append(record)
            print(seed, record, flush=True)
    (out / f"training_{seed}.json").write_text(json.dumps({
        "source": str(path.relative_to(HERE)), "training_seeds": SEEDS, "scales": SCALES,
        "additional_reset_seeds": [23200, 23295] if wide_resets else None,
        "prefix_samples": len(prefix[0]), "history": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    parser.add_argument("--wide-resets", action="store_true")
    parser.add_argument("--polish", action="store_true")
    args = parser.parse_args()
    train(args.seed, args.wide_resets, args.polish)
