"""Repair the prefix-trained actor's fixed-delay slalom via its delay input weights."""
import argparse
import json

import numpy as np
from stable_baselines3 import PPO
import torch

from distill import HERE, POLICY_SEEDS
from batch_sim import build_cases
from gauss_newton_actor import ErrorRollout

SEEDS = list(range(22730, 22738))
SCALES = [(.89, 1.03), (1.16, .97)]
CHECKPOINTS = (4, 10, 20)


def train(seed, source_step, stress=False, transient=False, extend_stress=False):
    torch.set_num_threads(1)
    source = HERE / "anchor_prefix_focus" / f"policy_{seed}_{source_step}.zip"
    if (stress or transient) and source_step == 900:
        source = HERE / "repair_fixed_slalom_900" / f"policy_{seed}_20.zip"
    if extend_stress:
        source = HERE / f"repair_fixed_slalom_{source_step}_stress" / f"policy_{seed}_20.zip"
    model = PPO.load(source, device="cpu")
    first = model.policy.mlp_extractor.policy_net[0]
    layer = model.policy.action_net
    original = np.r_[layer.weight.detach().numpy().ravel(), layer.bias.detach().numpy(),
                     first.bias.detach().numpy(), first.weight.detach().numpy()[:, 9:11].ravel()]
    parameters = original.copy()
    indices = np.arange(129, len(parameters))
    seeds = list(range(22750, 22758)) if stress else SEEDS
    scales = [(.88, .95), (1.11, 1.08)] if stress else SCALES
    if transient:
        seeds = list(range(22780, 22788))
        scales = [(.89, .97), (1.10, 1.06)]
    if extend_stress:
        seeds = list(range(22788, 22796))
        scales = [(.89, .99), (1.12, 1.06)]
    cases = [c for c in build_cases(seeds, scales) if c["delay"] != "none"]
    if stress:
        cases.extend(c for c in build_cases(seeds, [(1.19, 1.12), (1.21, 1.15)])
                     if c["delay"] == "fixed200" and c["path"] == "slalom")
    if transient:
        cases.extend(c for c in build_cases(seeds, [(1.13, 1.08), (1.17, 1.11)])
                     if c["delay"] == "fixed200" and c["path"] == "slalom")
    if extend_stress:
        cases.extend(c for c in build_cases(seeds, [(1.15, 1.085), (1.19, 1.11)])
                     if c["delay"] == "fixed200" and c["path"] == "slalom")
    fit = ErrorRollout(model.policy, cases)
    slalom = np.array([c["path"] == "slalom" and c["delay"] == "fixed200" for c in cases])
    slalom_condition = np.any(fit.groups[:, slalom], axis=1)
    baseline = fit.evaluate(parameters)
    ceiling = baseline["conditions"][~slalom_condition] * 1.008
    baseline_other = int(baseline["completion_by_episode"][~slalom].sum())
    damping = .001
    output = HERE / f"repair_fixed_slalom_{source_step}{'_stress' if stress else ''}"
    if transient:
        output = HERE / f"repair_fixed_slalom_{source_step}_transient"
    if extend_stress:
        output = HERE / f"repair_fixed_slalom_{source_step}_extended"
    output.mkdir(exist_ok=True)
    records = []
    def objective(result):
        full = result["rmse"][slalom].mean()
        if extend_stress:
            worst = np.sort(result["rmse"][slalom])[-max(1, int(slalom.sum()) // 4):]
            return .4 * full + .6 * worst.mean()
        if not transient:
            return full
        prefix = np.sqrt(np.mean(result["errors"][:201, slalom]**2, axis=0)).mean()
        return prefix + .15 * full
    phase_weights = np.ones_like(baseline["errors"])
    if transient:
        phase_weights[:201, slalom] *= 8.
        phase_weights[201:, slalom] *= .4
    for step in range(1, CHECKPOINTS[-1] + 1):
        current = fit.evaluate(parameters, derivatives=True, derivative_indices=indices)
        episode_weights = np.where(slalom, 12., 1.) / np.maximum(current["rmse"], 1e-7)
        if extend_stress:
            normalized = current["rmse"][slalom] / max(np.median(current["rmse"][slalom]), 1e-7)
            episode_weights[slalom] *= np.clip(normalized, .5, 4.)
        scale = np.sqrt(phase_weights * episode_weights[None, :])
        error = (current["errors"] * scale).ravel()
        jacobian = (current["gradient"] * scale[:, :, None]).reshape(-1, len(indices))
        hessian = jacobian.T @ jacobian
        direction = np.linalg.solve(
            hessian + np.diag(damping * np.maximum(hessian.diagonal(), 1e-3)), jacobian.T @ error
        )
        direction *= min(1., .18 / max(np.linalg.norm(direction), 1e-12))
        accepted = False
        for shrink in range(8):
            proposal = parameters.copy()
            proposal[indices] -= .5**shrink * direction
            result = fit.evaluate(proposal)
            better = objective(result) < objective(current)
            guarded = (
                result["completion_by_episode"][slalom].sum() >= current["completion_by_episode"][slalom].sum()
                and result["completion_by_episode"][~slalom].sum() >= baseline_other
                and np.all(result["conditions"][~slalom_condition] <= ceiling)
            )
            if better and guarded:
                parameters = proposal
                damping = max(1e-6, .8 * damping)
                accepted = True
                break
        if not accepted:
            result = current
            damping *= 5.
        if step in CHECKPOINTS:
            assert np.array_equal(parameters[:129], original[:129])
            with torch.no_grad():
                first.weight[:, 9:11].copy_(torch.tensor(parameters[indices].reshape(64, 2)))
            model.save(output / f"policy_{seed}_{step}.zip")
            record = {"step": step, "accepted": accepted, "slalom_rmse": float(result["rmse"][slalom].mean()),
                      "slalom_completed": int(result["completion_by_episode"][slalom].sum()),
                      "slalom_episodes": int(slalom.sum()), "completed": result["completed"]}
            records.append(record)
            print(seed, record, flush=True)
    (output / f"training_{seed}.json").write_text(json.dumps({
        "source": str(source.relative_to(HERE)), "training_seeds": seeds, "scales": scales,
        "additional_high_rate_slalom_cases": stress or transient or extend_stress,
        "emphasizes_first_two_seconds": transient,
        "emphasizes_highest_slalom_rmse": extend_stress,
        "baseline_slalom_rmse": float(baseline["rmse"][slalom].mean()), "snapshots": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    parser.add_argument("--source-step", type=int, choices=(450, 900), default=900)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--transient", action="store_true")
    parser.add_argument("--extend-stress", action="store_true")
    args = parser.parse_args()
    train(args.seed, args.source_step, args.stress, args.transient, args.extend_stress)
