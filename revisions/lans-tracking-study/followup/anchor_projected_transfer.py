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


def collect(prior, teacher, seeds, scales, prefix_focus=False, slalom_anchor=False):
    nominal, labels, delayed, importance, critical = [], [], [], [], []
    if prefix_focus:
        from receding_nominal_targets import SteeringTeacher
    for case in build_cases(seeds, scales):
        env = make_env(case)
        obs, _ = env.reset(seed=case["seed"])
        short_teacher = SteeringTeacher(env) if prefix_focus and case["delay"] == "none" else None
        for tick in range(1000):
            action = prior(obs)
            if tick < 80 or tick % 8 == 0:
                if case["delay"] == "none":
                    nominal.append(obs)
                    if short_teacher is None:
                        label = teacher(obs)
                    elif tick < 12:
                        label = short_teacher.action(env, tick) + env.K @ env.x
                    else:
                        label = action
                    labels.append(np.clip(label, -1., 1.))
                    importance.append(12. if prefix_focus and tick < 3 else
                                      4. if prefix_focus and tick < 12 else 1.)
                elif tick > 0:
                    if slalom_anchor and case["path"] == "slalom" and case["delay"] == "fixed200" and tick < 620:
                        critical.append(len(delayed))
                    delayed.append(obs)
            obs, _, terminated, truncated, _ = env.step([action])
            if terminated or truncated:
                break
    if slalom_anchor:
        for case in build_cases(range(23300, 23348), scales):
            if case["delay"] != "none" or case["mass"] != 1650:
                continue
            env = make_env(case)
            short_teacher = SteeringTeacher(env)
            obs, _ = env.reset(seed=case["seed"])
            for tick in range(8):
                command = short_teacher.action(env, tick) + float(env.K @ env.x)
                nominal.append(obs)
                labels.append(np.clip(command, -1., 1.))
                importance.append(12. if tick < 3 else 4.)
                obs, _, terminated, _, _ = env.step([prior(obs)])
                if terminated:
                    break
    return (
        torch.tensor(np.stack(nominal)),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(np.stack(delayed)),
        torch.tensor(importance),
        torch.tensor(critical, dtype=torch.long),
    )


def train(seed, correct_anchor=False, refresh_anchors=False, prefix_focus=False, slalom_anchor=False):
    prefix_focus = prefix_focus or slalom_anchor
    correct_anchor = correct_anchor or refresh_anchors or prefix_focus
    torch.set_num_threads(1)
    source = HERE / "guarded_round2" / "blend_.1" / f"policy_{seed}.zip"
    expert = HERE / "direct_fifo_nominal_polish" / f"policy_{seed}_48.zip"
    start = HERE / "anchor_corrected_transfer" / f"policy_{seed}_300.zip" if refresh_anchors else source
    seeds = [22592, 22593, 22594, 22595] if refresh_anchors else TRAIN_SEEDS
    scales = [(.88, 1.04), (1.13, .97)] if refresh_anchors else SCALES
    if prefix_focus:
        start = source = HERE / "gauss_newton_joint" / f"policy_{seed}_12.zip"
        seeds = list(range(22724, 22730))
        scales = [(.88, 1.03), (1.14, .96)]
    if slalom_anchor:
        start = source = HERE / "anchor_prefix_focus" / f"policy_{seed}_150.zip"
        seeds = list(range(23348, 23356))
        scales = [(.90, .96), (1.14, 1.08)]
    model = PPO.load(start, device="cpu")
    teacher = None if prefix_focus else actor(PPO.load(expert, device="cpu"))
    nominal, target, delayed, importance, critical = collect(
        actor(model), teacher, seeds, scales, prefix_focus, slalom_anchor
    )
    network = torch.nn.Sequential(model.policy.mlp_extractor.policy_net, model.policy.action_net)
    named = dict(network.named_parameters())
    parameters = tuple(named.values())

    def steering(weights, obs):
        return functional_call(network, weights, (obs,)).squeeze(-1)

    generator = torch.Generator().manual_seed(seed)
    output = HERE / ("anchor_corrected_transfer" if correct_anchor else "anchor_projected_transfer")
    if refresh_anchors:
        output = HERE / "anchor_refreshed_transfer"
    if prefix_focus:
        output = HERE / "anchor_prefix_focus"
    if slalom_anchor:
        output = HERE / "anchor_prefix_slalom"
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
    if prefix_focus:
        checkpoints = (150, 450, 900)
    if slalom_anchor:
        checkpoints = (80, 230, 480)
    for step in range(1, checkpoints[-1] + 1):
        sample = torch.randint(len(delayed), (96,), generator=generator)
        if slalom_anchor:
            additional = critical[torch.randint(len(critical), (48,), generator=generator)]
            sample = torch.cat((sample, additional))
        anchor = delayed[sample]
        jacobian = vmap(jacrev(steering), in_dims=(None, 0))(named, anchor)
        rows = torch.cat([matrix.reshape(len(anchor), -1) for matrix in jacobian.values()], dim=1).detach()
        values, vectors = torch.linalg.eigh(rows @ rows.T)
        keep = values > max(1e-7, float(values[-1]) * 1e-6)
        basis = (vectors[:, keep].T @ rows) / values[keep].sqrt()[:, None]

        index = torch.randint(len(nominal), (512,), generator=generator)
        if prefix_focus:
            index = torch.multinomial(importance, 512, replacement=True, generator=generator)
        prediction = network(nominal[index]).squeeze(-1)
        loss = torch.nn.functional.smooth_l1_loss(prediction, target[index], beta=.015)
        gradient = torch.cat([g.flatten() for g in torch.autograd.grad(loss, parameters)])
        direction = gradient - basis.T @ (basis @ gradient)
        step_size = .00065 if slalom_anchor else .001
        direction = direction * (step_size / direction.norm().clamp_min(1e-12))
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
        "nominal_teacher": "bounded receding targets" if prefix_focus else str(expert.relative_to(HERE)),
        "training_seeds": seeds,
        "extra_nominal_reset_seeds": [23300, 23347] if slalom_anchor else None,
        "fixed_slalom_anchor_samples": int(len(critical)),
        "corrects_cumulative_anchor_drift": correct_anchor,
        "scales": scales,
        "snapshots": snapshots,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    parser.add_argument("--correct-anchor", action="store_true")
    parser.add_argument("--refresh-anchors", action="store_true")
    parser.add_argument("--prefix-focus", action="store_true")
    parser.add_argument("--slalom-anchor", action="store_true")
    args = parser.parse_args()
    train(args.seed, args.correct_anchor, args.refresh_anchors, args.prefix_focus, args.slalom_anchor)
