"""Fit the existing actor output by propagating steering sensitivities through the FIFO."""
import argparse
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from stable_baselines3 import PPO
import torch

from distill import HERE, POLICY_SEEDS
from batch_sim import build_cases, prepare
from core import KR, XS, beta
from features import whiten

TRAIN_SEEDS = [22600, 22601, 22602, 22603, 22604, 22605]
SCALES = [(.90, 1.04), (1.12, .96)]
CHECKPOINTS = (4, 12, 24)
STEPS = 1000


class ErrorRollout:
    def __init__(self, policy, cases):
        self.cases = cases
        self.data = prepare(cases)
        self.layers = [p.detach().numpy().copy()
                       for p in policy.mlp_extractor.policy_net.parameters()]
        self.nominal = np.array([c["delay"] == "none" for c in cases])
        keys = [(c["path"], c["delay"], c["mass"]) for c in cases]
        conditions = list(dict.fromkeys(keys))
        self.groups = np.array([[key == condition for key in keys] for condition in conditions])
        self.delayed = np.array([c[1] != "none" for c in conditions])
        self.steps = .01 * np.arange(1, 5)
        self.episodes = np.arange(len(cases))
        delay = self.data["delays"]
        self.gain = beta(delay[:, :STEPS])
        self.previous = np.pad(delay[:, :STEPS-1], ((0, 0), (1, 0))) / 200
        self.history = np.zeros_like(self.previous)
        for tick in range(1, STEPS):
            self.history[:, tick] = delay[:, max(0, tick-10):tick].mean(axis=1) / 200
        ref = self.data["ref"]
        self.disturbance = (
            self.data["G"][:, :, None] * ref[:, None, 2, :STEPS]
            + self.data["H"][:, :, None] * ref[:, None, 3, :STEPS]
        ) / 25

    def evaluate(self, parameters, derivatives=False, derivative_indices=None):
        w1, b1, w2, b2 = self.layers
        head = parameters[:65]
        fit_bias = len(parameters) > len(head)
        fit_delay = len(parameters) > 129
        if fit_bias:
            b1 = parameters[65:129]
        if fit_delay:
            w1 = np.column_stack((w1[:, :9], parameters[129:].reshape(64, 2)))
        ref = self.data["ref"]
        state = self.data["initial"].copy()
        count = len(state)
        active = np.ones(count, dtype=bool)
        commands = np.zeros((STEPS, count))
        errors = np.zeros((STEPS + 1, count))
        errors[0] = state[:, 0]
        dimension = len(parameters) if derivative_indices is None else len(derivative_indices)
        state_gradient = np.zeros((count, 4, dimension)) if derivatives else None
        command_gradient = np.zeros((STEPS, count, dimension)) if derivatives else None
        error_gradient = np.zeros((STEPS + 1, count, dimension)) if derivatives else None

        for tick in range(STEPS):
            offset = self.steps * (25 * state[:, 2, None])
            preview = (
                ref[:, 0, tick, None] + state[:, 0, None]
                + (ref[:, 1, tick, None] + 25 * state[:, 2, None]) * self.steps
                - ref[:, 0, tick + np.arange(1, 5)]
            ).astype(np.float32).astype(float)
            channels = (preview - state[:, 0, None] - offset) / (10 * self.steps**2)
            obs = whiten(np.column_stack([
                state / XS, np.ones(count), channels,
                self.previous[:, tick], self.history[:, tick],
            ]).astype(np.float32))
            z1 = obs @ w1.T + b1
            z2 = np.maximum(z1, 0) @ w2.T + b2
            hidden = np.maximum(z2, 0)
            raw = (hidden @ head[:-1]) + head[-1]
            total = self.gain[:, tick] * np.clip(raw, -1, 1) - state @ KR
            commands[tick] = np.clip(total, -.5, .5)
            generation = self.data["applied"][:, tick]
            previous = np.maximum(0, generation)
            applied = commands[previous, self.episodes] * (generation >= 0)
            if derivatives:
                # Preview subtraction cancels analytic state derivatives. The cast
                # to float32 is treated straight-through, as in the Torch trainer.
                bias_slope = ((z2 > 0) * head[:-1]) @ w2 * (z1 > 0)
                raw_state = bias_slope @ w1[:, :4] / XS
                if fit_delay:
                    delay_slope = (bias_slope[:, :, None] * obs[:, None, 9:11]).reshape(count, -1)
                    gradient = np.column_stack((hidden, np.ones(count), bias_slope, delay_slope))
                elif fit_bias:
                    gradient = np.column_stack((hidden, np.ones(count), bias_slope))
                else:
                    gradient = np.column_stack((hidden, np.ones(count)))
                if derivative_indices is not None:
                    gradient = gradient[:, derivative_indices]
                gradient += np.einsum("ni,nij->nj", raw_state, state_gradient)
                gradient *= (self.gain[:, tick] * (np.abs(raw) < 1))[:, None]
                gradient -= np.einsum("i,nij->nj", KR, state_gradient)
                command_gradient[tick] = gradient * (np.abs(total) < .5)[:, None]
                applied_gradient = command_gradient[previous, self.episodes] * (generation >= 0)[:, None]
                next_gradient = (
                    self.data["A"] @ state_gradient
                    + self.data["B"][:, :, None] * applied_gradient[:, None, :]
                )
                state_gradient[active] = next_gradient[active]
                error_gradient[tick + 1, active] = state_gradient[active, 0]
                excess = ~active & (np.abs(state[:, 0]) > 2.5)
                error_gradient[tick + 1, excess] = (
                    np.sign(state[excess, 0])[:, None] * state_gradient[excess, 0]
                )
            new_state = (
                np.einsum("nij,nj->ni", self.data["A"], state)
                + self.data["B"] * applied[:, None] + self.disturbance[:, :, tick]
            )
            state[active] = new_state[active]
            errors[tick + 1] = np.where(active, state[:, 0], np.maximum(2.5, abs(state[:, 0])))
            active &= (abs(state[:, 0]) <= 2.5) & (abs(state[:, 2]) <= .5)

        rmse = np.sqrt((errors**2).mean(axis=0))
        return {
            "mean_nominal": float(rmse[self.nominal].mean()) if self.nominal.any() else None,
            "conditions": self.groups @ rmse / self.groups.sum(axis=1),
            "completed": int(active.sum()),
            "completion_by_episode": active,
            "errors": errors,
            "gradient": error_gradient,
            "rmse": rmse,
        }


def train(seed, joint, phase_weighted=False, margin_guard=False, protected_conditions=False, fit_bias=False, fit_delay=False):
    fit_bias = fit_bias or fit_delay
    protected_conditions = protected_conditions or fit_bias
    margin_guard = margin_guard or protected_conditions
    phase_weighted = phase_weighted or margin_guard
    joint = joint or phase_weighted
    torch.set_num_threads(1)
    if joint:
        source = HERE / "anchor_closed_loop" / f"policy_{seed}_24.zip"
        output = HERE / "gauss_newton_joint"
    else:
        source = HERE / "direct_fifo_nominal_polish" / f"policy_{seed}_48.zip"
        output = HERE / "gauss_newton_nominal"
    if phase_weighted:
        source = HERE / "gauss_newton_joint" / f"policy_{seed}_12.zip"
        output = HERE / "gauss_newton_phase"
    if margin_guard:
        output = HERE / "gauss_newton_margin"
    if protected_conditions:
        output = HERE / "gauss_newton_protected"
    if fit_bias:
        source = HERE / "gauss_newton_margin" / f"policy_{seed}_24.zip"
        output = HERE / "gauss_newton_bias"
    if fit_delay:
        source = HERE / "gauss_newton_bias" / f"policy_{seed}_12.zip"
        output = HERE / "gauss_newton_delay_columns"
    output.mkdir(exist_ok=True)
    model = PPO.load(source, device="cpu")
    layer = model.policy.action_net
    head = np.r_[layer.weight.detach().numpy().ravel(), layer.bias.detach().numpy()]
    seeds = list(range(22630, 22638)) if phase_weighted else TRAIN_SEEDS
    scales = [(.88, 1.03), (1.15, .97)] if phase_weighted else SCALES
    if margin_guard:
        seeds = list(range(22650, 22660))
        scales = [(.87, .99), (1.14, 1.08)]
    if protected_conditions:
        seeds = list(range(22660, 22668))
        scales = [(.91, .94), (1.12, 1.06)]
    if fit_bias:
        seeds = list(range(22702, 22710))
        scales = [(.89, .97), (1.16, 1.04)]
        head = np.r_[head, model.policy.mlp_extractor.policy_net[0].bias.detach().numpy()]
    if fit_delay:
        seeds = list(range(22710, 22716))
        scales = [(.91, 1.01), (1.13, .96)]
        head = np.r_[head, model.policy.mlp_extractor.policy_net[0].weight[:, 9:11].detach().numpy().ravel()]
    cases = build_cases(seeds, scales)
    if not joint:
        cases = [case for case in cases if case["delay"] == "none"]
    rollout = ErrorRollout(model.policy, cases)
    baseline = rollout.evaluate(head)
    phase_weights = np.ones((STEPS + 1, len(cases)))
    if phase_weighted:
        for index, case in enumerate(cases):
            if case["delay"] != "none":
                continue
            if case["mass"] < 1650:
                phase_weights[41:, index] *= 4.
            elif case["mass"] > 1650:
                phase_weights[41:, index] *= 2.
            else:
                phase_weights[1:11, index] *= 3.
    ceiling = baseline["conditions"].copy()
    if margin_guard:
        old = PPO.load(HERE / "guarded_round2" / "blend_.1" / f"policy_{seed}.zip", device="cpu")
        old_layer = old.policy.action_net
        old_head = np.r_[old_layer.weight.detach().numpy().ravel(), old_layer.bias.detach().numpy()]
        old_errors = ErrorRollout(old.policy, cases).evaluate(old_head)
        ceiling = np.maximum(ceiling, old_errors["conditions"])
    ceiling *= 1.005
    protected = np.ones_like(rollout.delayed) if protected_conditions else rollout.delayed
    if protected_conditions:
        ceiling[~rollout.delayed] = baseline["conditions"][~rollout.delayed] * 1.0002
    observations = []
    damping = .0001
    checkpoints = (3, 8) if fit_delay else ((4, 12) if fit_bias else CHECKPOINTS)
    for step in range(1, checkpoints[-1] + 1):
        current = rollout.evaluate(head, derivatives=True)
        sample_weight = 1 / np.maximum(current["rmse"], 1e-7)
        if margin_guard:
            sample_weight[~rollout.nominal] = 0.
        elif joint:
            sample_weight[~rollout.nominal] *= 2.
        weight = np.sqrt(phase_weights * sample_weight[None, :])
        residual = (current["errors"] * weight).ravel()
        jacobian = (current["gradient"] * weight[:, :, None]).reshape(-1, len(head))
        hessian = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        diagonal = np.maximum(hessian.diagonal(), 1e-3)
        regularized = hessian + np.diag(damping * diagonal)
        if margin_guard:
            # Linearized change for the actual mean episode metric in each delayed condition.
            episode_slope = np.einsum("kn,knp->np", current["errors"], current["gradient"])
            episode_slope /= (1001 * np.maximum(current["rmse"], 1e-7))[:, None]
            groups = rollout.groups[protected]
            cap = ceiling[protected]
            condition_slope = (groups @ episode_slope) / groups.sum(axis=1)[:, None]
            remaining = 1. - current["conditions"][protected] / cap
            parameter_scale = np.sqrt(diagonal)
            scaled_hessian = regularized / np.outer(parameter_scale, parameter_scale)
            scaled_gradient = gradient / parameter_scale
            limits = .008 * parameter_scale
            constraint = LinearConstraint(-condition_slope / cap[:, None] / parameter_scale,
                                          -np.inf, remaining)
            fit = minimize(
                lambda d: .5 * d @ scaled_hessian @ d - scaled_gradient @ d,
                np.zeros(len(head)),
                jac=lambda d: scaled_hessian @ d - scaled_gradient,
                bounds=Bounds(-limits, limits), constraints=[constraint],
                method="SLSQP", options={"maxiter": 80, "ftol": 1e-9},
            )
            direction = fit.x / parameter_scale
        else:
            direction = np.linalg.solve(regularized, gradient)
        max_step = .05 if margin_guard else (.015 if joint else .08)
        direction *= min(1., max_step / max(np.linalg.norm(direction), 1e-12))
        accepted = False
        for shrink in range(7):
            trial = (head - .5**shrink * direction).astype(np.float32)
            result = rollout.evaluate(trial)
            guard = np.all(result["conditions"][protected] <= ceiling[protected])
            if (result["mean_nominal"] < current["mean_nominal"]
                    and result["completed"] >= baseline["completed"] and guard):
                head = trial
                damping = max(1e-6, damping * .8)
                accepted = True
                break
        if not accepted:
            damping *= 8
            result = current
        if step in checkpoints:
            with torch.no_grad():
                layer.weight.copy_(torch.tensor(head[:64][None, :]))
                layer.bias.copy_(torch.tensor(head[64:65]))
                if fit_bias:
                    model.policy.mlp_extractor.policy_net[0].bias.copy_(torch.tensor(head[65:129]))
                if fit_delay:
                    model.policy.mlp_extractor.policy_net[0].weight[:, 9:11].copy_(
                        torch.tensor(head[129:].reshape(64, 2))
                    )
            model.save(output / f"policy_{seed}_{step}.zip")
            record = {"step": step, "nominal_rmse": result["mean_nominal"],
                      "completed": result["completed"], "accepted": accepted, "damping": damping}
            observations.append(record)
            print(seed, record, flush=True)
    (output / f"training_{seed}.json").write_text(json.dumps({
        "source": str(source.relative_to(HERE)),
        "training_seeds": seeds,
        "scales": scales,
        "phase_weighted": phase_weighted,
        "margin_guard": margin_guard,
        "protects_nominal_conditions": protected_conditions,
        "fits_first_layer_bias": fit_bias,
        "fits_delay_columns": fit_delay,
        "baseline_nominal_rmse": baseline["mean_nominal"],
        "snapshots": observations,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--phase-weighted", action="store_true")
    parser.add_argument("--margin-guard", action="store_true")
    parser.add_argument("--protected-conditions", action="store_true")
    parser.add_argument("--fit-bias", action="store_true")
    parser.add_argument("--fit-delay", action="store_true")
    args = parser.parse_args()
    train(args.seed, args.joint, args.phase_weighted, args.margin_guard, args.protected_conditions, args.fit_bias, args.fit_delay)
