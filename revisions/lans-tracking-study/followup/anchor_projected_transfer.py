"""Transfer nominal targets in directions that preserve sampled delayed outputs."""
import argparse
import json

import numpy as np
from stable_baselines3 import PPO
import torch
from torch.func import functional_call, jacrev, vmap

from distill import HERE, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases

TRAIN_SEEDS = [22580, 22581, 22582, 22583]
SCALES = [(.91, .97), (1.14, 1.06)]
CHECKPOINTS = (30, 120, 300)


def collect(prior, teacher, seeds, scales):
    nominal, labels, delayed = [], [], []
    for case in build_cases(seeds, scales):
        env = make_env(case)
        obs, _ = env.reset(seed=case["seed"])
        for tick in range(1000):
            action = prior(obs)
            if tick < 80 or tick % 8 == 0:
                if case["delay"] == "none":
                    nominal.append(obs)
                    labels.append(np.clip(teacher(obs), -1., 1.))
                elif tick > 0:
                    delayed.append(obs)
            obs, _, terminated, truncated, _ = env.step([action])
            if terminated or truncated:
                break
    return (
        torch.tensor(np.stack(nominal)),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(np.stack(delayed)),
    )


def train(seed, correct_anchor=False, refresh_anchors=False):
    correct_anchor = correct_anchor or refresh_anchors
    torch.set_num_threads(1)
    source = HERE / "guarded_round2" / "blend_.1" / f"policy_{seed}.zip"
    expert = HERE / "direct_fifo_nominal_polish" / f"policy_{seed}_48.zip"
    start = HERE / "anchor_corrected_transfer" / f"policy_{seed}_300.zip" if refresh_anchors else source
    seeds = [22592, 22593, 22594, 22595] if refresh_anchors else TRAIN_SEEDS
    scales = [(.88, 1.04), (1.13, .97)] if refresh_anchors else SCALES
    model = PPO.load(start, device="cpu")
    nominal, target, delayed = collect(actor(model), actor(PPO.load(expert, device="cpu")), seeds, scales)
    network = torch.nn.Sequential(model.policy.mlp_extractor.policy_net, model.policy.action_net)
    named = dict(network.named_parameters())
    parameters = tuple(named.values())

    def steering(weights, obs):
        return functional_call(network, weights, (obs,)).squeeze(-1)

    generator = torch.Generator().manual_seed(seed)
    output = HERE / ("anchor_corrected_transfer" if correct_anchor else "anchor_projected_transfer")
    if refresh_anchors:
        output = HERE / "anchor_refreshed_transfer"
    output.mkdir(exist_ok=True)
    snapshots = []
    with torch.no_grad():
        if refresh_anchors:
            baseline = PPO.load(source, device="cpu")
            reference_output = baseline.policy.action_net(
                baseline.policy.mlp_extractor.policy_net(delayed)
            ).squeeze(-1)
        else:
            reference_output = network(delayed).squeeze(-1) if correct_anchor else None
    checkpoints = (40, 120, 300) if correct_anchor else CHECKPOINTS
    if refresh_anchors:
        checkpoints = (120, 300, 900)
    for step in range(1, checkpoints[-1] + 1):
        sample = torch.randint(len(delayed), (96,), generator=generator)
        anchor = delayed[sample]
        jacobian = vmap(jacrev(steering), in_dims=(None, 0))(named, anchor)
        rows = torch.cat([matrix.reshape(len(anchor), -1) for matrix in jacobian.values()], dim=1).detach()
        values, vectors = torch.linalg.eigh(rows @ rows.T)
        keep = values > max(1e-7, float(values[-1]) * 1e-6)
        basis = (vectors[:, keep].T @ rows) / values[keep].sqrt()[:, None]

        index = torch.randint(len(nominal), (512,), generator=generator)
        prediction = network(nominal[index]).squeeze(-1)
        loss = torch.nn.functional.smooth_l1_loss(prediction, target[index], beta=.015)
        gradient = torch.cat([g.flatten() for g in torch.autograd.grad(loss, parameters)])
        direction = gradient - basis.T @ (basis @ gradient)
        direction = direction * (.001 / direction.norm().clamp_min(1e-12))
        if correct_anchor:
            with torch.no_grad():
                drift_from_source = network(anchor).squeeze(-1) - reference_output[sample]
            correction = basis.T @ ((vectors[:, keep].T @ drift_from_source) / values[keep].sqrt())
            scale = min(.25, .0007 / max(float(correction.norm()), 1e-12))
            direction = direction + scale * correction
        updates = direction.split([p.numel() for p in parameters])
        with torch.no_grad():
            before = network(anchor).squeeze(-1)
            for parameter, update in zip(parameters, updates, strict=True):
                parameter.sub_(update.reshape_as(parameter))
            drift = float((network(anchor).squeeze(-1) - before).abs().max())
        if step in checkpoints:
            model.save(output / f"policy_{seed}_{step}.zip")
            record = {"step": step, "nominal_target_loss": float(loss.detach()),
                      "max_local_anchor_drift": drift, "anchor_rank": int(keep.sum())}
            snapshots.append(record)
            print(seed, record, flush=True)
    (output / f"training_{seed}.json").write_text(json.dumps({
        "source": str(start.relative_to(HERE)),
        "delayed_anchor": str(source.relative_to(HERE)),
        "nominal_teacher": str(expert.relative_to(HERE)),
        "training_seeds": seeds,
        "corrects_cumulative_anchor_drift": correct_anchor,
        "scales": scales,
        "snapshots": snapshots,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    parser.add_argument("--correct-anchor", action="store_true")
    parser.add_argument("--refresh-anchors", action="store_true")
    args = parser.parse_args()
    train(args.seed, args.correct_anchor, args.refresh_anchors)
