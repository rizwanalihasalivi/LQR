"""Optimize early tracking with closed-loop nominal and FIFO-delayed rollouts."""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
import torch

from distill import HERE, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases, prepare
from core import KR, XS, beta
from features import T

HORIZON = 120
TRAINING_SEEDS = [22522, 22523, 22524]
SCALES = [(.88, 1.02), (1.15, .98)]


def main(seed, full_episode=False, nominal_priority=False, projected=False, full_actor=False, line_search=False, condition_guard=False, project_conditions=False, unconstrained_nominal=False, nominal_continue=False, nominal_polish=False, nominal_rmse=False):
    nominal_polish = nominal_polish or nominal_rmse
    unconstrained_nominal = unconstrained_nominal or nominal_continue or nominal_polish
    condition_guard = condition_guard or project_conditions
    line_search = line_search or condition_guard
    full_actor = full_actor or line_search
    projected = projected or full_actor
    full_episode = full_episode or nominal_priority or projected or unconstrained_nominal
    horizon = 1000 if full_episode else HORIZON
    torch.set_num_threads(1)
    source = HERE / "guarded_round2" / "blend_.1" / f"policy_{seed}.zip"
    if nominal_continue:
        source = HERE / "direct_fifo_nominal_only" / f"policy_{seed}_64.zip"
    if nominal_polish:
        source = HERE / "nominal_oracle_actor" / f"policy_{seed}_2400.zip"
    if nominal_rmse:
        source = HERE / "direct_fifo_nominal_polish" / f"policy_{seed}_48.zip"
    model = PPO.load(source, device="cpu")
    predict = actor(model)
    seeds = [22525, 22526, 22527] if full_episode else TRAINING_SEEDS
    scales = [(.91, 1.05), (1.12, .95)] if full_episode else SCALES
    if condition_guard:
        seeds = [22528, 22529, 22530, 22531, 22532]
        scales = [(.87, 1.08), (1.13, .94)]
    if unconstrained_nominal:
        seeds = [22533, 22534, 22535, 22536, 22537]
        scales = [(.92, .89), (1.12, 1.11)]
    if nominal_continue:
        seeds = [22538, 22539, 22540, 22541, 22542]
        scales = [(.89, .97), (1.13, 1.05)]
    if nominal_polish:
        seeds = [22558, 22559, 22560, 22561, 22562, 22563]
        scales = [(.88, 1.02), (1.16, .95)]
    if nominal_rmse:
        seeds = [22569, 22570, 22571, 22572, 22573, 22574]
        scales = [(.89, .96), (1.13, 1.07)]
    cases = build_cases(seeds, scales)
    if unconstrained_nominal:
        cases = [case for case in cases if case["delay"] == "none"]
    data = prepare(cases)
    transition, steer, initial = (torch.tensor(data[k]) for k in ("A", "B", "initial"))
    refs = torch.tensor(data["ref"])
    disturbances = (
        data["G"][:, :, None] * data["ref"][:, None, 2, :horizon]
        + data["H"][:, :, None] * data["ref"][:, None, 3, :horizon]
    ) / 25
    disturbances = torch.tensor(disturbances.transpose(2, 0, 1))
    scale = torch.tensor(XS)
    gain = torch.tensor(KR)
    whitening = torch.tensor(T)
    horizons = torch.arange(1, 5, dtype=torch.float64) * .01
    speed = torch.ones((len(cases), 1), dtype=torch.float64)
    observed = np.zeros((horizon, len(cases), 2))
    for tick in range(1, horizon):
        observed[tick, :, 0] = data["delays"][:, tick - 1] / 200
        observed[tick, :, 1] = data["delays"][:, max(0, tick - 10):tick].mean(axis=1) / 200
    observed = torch.tensor(observed)
    decay = torch.tensor(beta(data["delays"][:, :horizon].T))
    generations = torch.tensor(data["applied"][:, :horizon].T + 1)
    window = int((torch.arange(horizon)[:, None] + 1 - generations).max()) + 1
    generation_offsets = generations - (torch.arange(horizon) + 2 - window).clamp_min(0)[:, None]
    episode_ids = torch.arange(len(cases))
    modes = torch.tensor([[case["delay"] == mode for case in cases]
                          for mode in ("none", "mixed", "fixed200")], dtype=torch.float64)
    if condition_guard:
        delayed_masks = torch.tensor([
            [case["delay"] == mode and case["path"] == path for case in cases]
            for mode in ("mixed", "fixed200")
            for path in ("dlc", "step", "sine", "slalom")
        ], dtype=torch.float64)
        modes = torch.cat((modes, delayed_masks))
    if nominal_polish:
        conditions = list(dict.fromkeys((case["path"], case["mass"]) for case in cases))
        nominal_masks = torch.tensor([
            [(case["path"], case["mass"]) == condition for case in cases]
            for condition in conditions
        ], dtype=torch.float64)
        modes = torch.cat((modes, nominal_masks))
    mode_weights = modes / modes.sum(dim=1, keepdim=True).clamp_min(1.)
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
    anchor_obs = torch.tensor(np.stack(observations)) if observations else None
    with torch.no_grad():
        if anchor_obs is not None:
            anchors = hidden(anchor_obs)
            anchor_actions = head(anchors).squeeze(-1).clone()

    def rollout():
        state = initial
        loss = torch.zeros(len(cases), dtype=torch.float64)
        commands = [torch.zeros_like(loss)]
        for tick in range(horizon):
            preview = (
                refs[:, 0, tick, None] + state[:, 0, None]
                + (refs[:, 1, tick, None] + 25 * state[:, 2, None]) * horizons
                - refs[:, 0, tick + 1:tick + 5]
            ).float().double()
            channels = (
                preview - state[:, 0, None] - 25 * state[:, 2, None] * horizons
            ) / (10 * horizons**2)
            conditioned = channels.float().double() @ whitening
            obs = torch.cat((state / scale, speed, conditioned, observed[tick]), dim=-1).float()
            command = (decay[tick] * head(hidden(obs)).squeeze(-1).double().clamp(-1, 1) - state @ gain)
            command = command.clamp(-.5, .5)
            commands.append(command)
            applied = torch.stack(commands[-window:])[generation_offsets[tick], episode_ids]
            state = torch.einsum("nij,nj->ni", transition, state) + steer * applied[:, None]
            state = state + disturbances[tick]
            loss = loss + state[:, 0].square()
        if nominal_rmse:
            return mode_weights @ ((loss + initial[:, 0].square()) / (horizon + 1)).sqrt()
        return mode_weights @ (loss / horizon)

    for parameter in hidden.parameters():
        parameter.requires_grad_(full_actor or unconstrained_nominal)
    if unconstrained_nominal:
        learning_rate = .00006 if nominal_rmse else (.000012 if nominal_polish else .00025)
        optimizer = torch.optim.Adam([*hidden.parameters(), *head.parameters()], lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(head.parameters(), lr=.00006 if nominal_priority else .000025)
    reference_loss = rollout().detach().clamp_min(1e-12)
    out = HERE / ("direct_fifo_full" if full_episode else "direct_fifo")
    if nominal_priority:
        out = HERE / "direct_fifo_nominal"
    if projected:
        out = HERE / "direct_fifo_projected"
    if full_actor:
        out = HERE / "direct_fifo_full_actor"
    if line_search:
        out = HERE / "direct_fifo_line_search"
    if condition_guard:
        out = HERE / "direct_fifo_condition_guard"
    if project_conditions:
        out = HERE / "direct_fifo_condition_projection"
    if unconstrained_nominal:
        out = HERE / "direct_fifo_nominal_only"
    if nominal_continue:
        out = HERE / "direct_fifo_nominal_extended"
    if nominal_polish:
        out = HERE / "direct_fifo_nominal_polish"
    if nominal_rmse:
        out = HERE / "direct_fifo_nominal_rmse"
    out.mkdir(exist_ok=True)
    checkpoints = (8, 20, 35) if full_episode else (25, 60, 100)
    if projected:
        checkpoints = (8, 16, 24)
    if condition_guard:
        checkpoints = (8, 16)
    if project_conditions:
        checkpoints = (5, 10)
    if unconstrained_nominal:
        checkpoints = (8, 24, 64)
    if nominal_continue:
        checkpoints = (24, 64)
    if nominal_polish:
        checkpoints = (8, 24, 48)
    if nominal_rmse:
        checkpoints = (24, 64, 128)
    weights = tuple(head.parameters())
    if full_actor:
        weights += tuple(hidden.parameters())
    accepted_steps = 0
    for step in range(1, checkpoints[-1] + 1):
        ratios = rollout() / reference_loss
        drift = torch.zeros(()) if unconstrained_nominal else (
            head(hidden(anchor_obs) if full_actor else anchors).squeeze(-1) - anchor_actions
        ).square().mean()
        loss = ratios.mean() + 2 * (ratios[1:] - 1).clamp_min(0).sum() + 5000 * drift
        if nominal_priority:
            delayed_regression = (ratios[1:] - 1).clamp_min(0)
            loss = ratios[3:].mean() if nominal_polish else ratios[0] + .25 * ratios[1] + 4 * delayed_regression.sum() + 2000 * drift
        if unconstrained_nominal:
            loss = ratios[3:].mean() if nominal_polish else ratios[0]
        if projected:
            constraints = ratios[3:] if project_conditions else ratios[1:3]
            delayed = torch.stack([
                torch.cat([g.flatten() for g in torch.autograd.grad(ratio, weights, retain_graph=True)])
                for ratio in constraints
            ])
            nominal_gradient = torch.cat([
                g.flatten() for g in torch.autograd.grad(ratios[0] + 2000 * drift, weights)
            ])
            basis = torch.linalg.svd(delayed, full_matrices=False).Vh
            direction = nominal_gradient - basis.T @ (basis @ nominal_gradient)
            direction = direction * ((.0012 if full_actor else .00022) / direction.norm().clamp_min(1e-12))
            updates = direction.split([p.numel() for p in weights])
            with torch.no_grad():
                if line_search:
                    original = tuple(p.clone() for p in weights)
                    baseline = ratios.detach()
                    for trial in range(8):
                        fraction = .5**trial
                        for parameter, previous, update in zip(weights, original, updates, strict=True):
                            parameter.copy_(previous - fraction * update.reshape_as(parameter))
                        measured = rollout() / reference_loss
                        guards = measured[3:] if condition_guard else measured[1:]
                        if measured[0] < baseline[0] and torch.all(guards <= 1.01):
                            accepted_steps += 1
                            break
                    else:
                        for parameter, previous in zip(weights, original, strict=True):
                            parameter.copy_(previous)
                else:
                    for parameter, update in zip(weights, updates, strict=True):
                        parameter.sub_(update.reshape_as(parameter))
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step in checkpoints:
            model.save(out / f"policy_{seed}_{step}.zip")
            print(seed, step, ratios.detach().tolist(), float(drift.detach()), accepted_steps, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    parser.add_argument("--full-episode", action="store_true")
    parser.add_argument("--nominal-priority", action="store_true")
    parser.add_argument("--projected", action="store_true")
    parser.add_argument("--full-actor", action="store_true")
    parser.add_argument("--line-search", action="store_true")
    parser.add_argument("--condition-guard", action="store_true")
    parser.add_argument("--project-conditions", action="store_true")
    parser.add_argument("--nominal-only", action="store_true")
    parser.add_argument("--nominal-continue", action="store_true")
    parser.add_argument("--nominal-polish", action="store_true")
    parser.add_argument("--nominal-rmse", action="store_true")
    args = parser.parse_args()
    main(args.seed, args.full_episode, args.nominal_priority, args.projected, args.full_actor, args.line_search, args.condition_guard, args.project_conditions, args.nominal_only, args.nominal_continue, args.nominal_polish, args.nominal_rmse)
