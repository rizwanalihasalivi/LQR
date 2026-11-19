"""Tune the existing LANS output heads on validation episodes only."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from stable_baselines3 import PPO

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "code"))
from batch_sim import build_cases, prepare
from core import KR, XS, beta
from features import whiten


def read_means(paths):
    scores = defaultdict(list)
    for path in paths:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                key = row["path"], row["delay"], float(row["mass"])
                scores[key].append(float(row["rmse_penalized"]))
    return {key: np.mean(values) for key, values in scores.items()}


def main():
    validation_mpc = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "balanced"
    if mode not in {"balanced", "nominal", "broad", "blend", "robust", "mixed", "transient", "initial", "tail", "recovery"}:
        raise ValueError(f"Unknown search mode: {mode}")
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    seeds = (72, 73, 74)
    source_paths = [
        HERE / ("initial_trial" if mode in {"tail", "recovery"} else "broad_trial") / f"policy_{seed}.zip"
        if mode in {"mixed", "transient", "initial", "tail", "recovery"}
        else STUDY / "models" / f"limited_lans_{seed}" / "policy.zip"
        for seed in seeds
    ]
    models = [PPO.load(path, device="cpu") for path in source_paths]
    layers = [
        [p.detach().numpy().copy() for p in m.policy.mlp_extractor.policy_net.parameters()]
        for m in models
    ]
    weights = np.stack([
        np.r_[m.policy.action_net.weight.detach().numpy().ravel(),
              m.policy.action_net.bias.detach().numpy()]
        for m in models
    ])
    if mode in {"broad", "blend", "robust", "mixed", "transient", "initial", "tail", "recovery"}:
        from calibration import SEEDS, SCALES
        base_cases = build_cases(SEEDS, SCALES)
    else:
        base_cases = build_cases([22000, 22001])
    case_keys = [(c["path"], c["delay"], c["mass"]) for c in base_cases]
    keys = list(dict.fromkeys(case_keys))
    condition_mask = np.array([[case == key for case in case_keys] for key in keys])
    condition_weights = condition_mask / condition_mask.sum(axis=1, keepdims=True)
    cases = base_cases * len(seeds)
    data = prepare(cases)
    count = len(base_cases)
    groups = [
        slice(i * count, (i + 1) * count) for i in range(len(seeds))
    ]
    nominal = np.array([key[1] == "none" for key in keys])
    mixed = np.array([key[1] == "mixed" for key in keys])
    fixed = ~(nominal | mixed)
    horizons = 0.01 * np.arange(1, 5)
    ref = data["ref"]
    delay = data["delays"]
    previous = np.pad(delay[:, :999], ((0, 0), (1, 0))) / 200
    history = np.zeros_like(previous)
    for tick in range(1, 1000):
        history[:, tick] = delay[:, max(0, tick - 10):tick].mean(axis=1) / 200
    gain = beta(delay[:, :1000])
    disturbance = (
        data["G"][:, :, None] * ref[:, None, 2, :1000]
        + data["H"][:, :, None] * ref[:, None, 3, :1000]
    ) / 25
    episode_indices = np.arange(len(cases))

    def rollout(heads, collect=False):
        state = data["initial"].copy()
        errors = state[:, 0] ** 2
        tail_errors = np.zeros(len(cases))
        lengths = np.zeros(len(cases), dtype=int)
        active = np.ones(len(cases), dtype=bool)
        commands = np.zeros((1000, len(cases)))
        samples = []
        for tick in range(1000):
            preview = (
                ref[:, 0, tick, None] + state[:, 0, None]
                + (ref[:, 1, tick, None] + 25 * state[:, 2, None]) * horizons
                - ref[:, 0, tick + np.arange(1, 5)]
            ).astype(np.float32).astype(float)
            channels = (
                preview - state[:, 0, None] - 25 * state[:, 2, None] * horizons
            ) / (10 * horizons**2)
            obs = whiten(np.column_stack([
                state / XS, np.ones(len(cases)), channels,
                previous[:, tick], history[:, tick],
            ]).astype(np.float32))
            features = np.empty((len(cases), 65), dtype=np.float32)
            features[:, -1] = 1
            residual = np.empty(len(cases))
            for i, group in enumerate(groups):
                w1, b1, w2, b2 = layers[i]
                hidden = np.maximum(obs[group] @ w1.T + b1, 0)
                features[group, :-1] = np.maximum(hidden @ w2.T + b2, 0)
                residual[group] = (features[group] * heads[i]).sum(axis=1)
            if collect and (tick < 75 or tick % 10 == 0):
                samples.append((features.copy(), obs.copy()))
            commands[tick] = np.clip(
                -state @ KR + gain[:, tick] * np.clip(residual, -1, 1), -.5, .5
            )
            generation = data["applied"][:, tick]
            applied = np.where(
                generation >= 0,
                commands[np.maximum(0, generation), episode_indices],
                0,
            )
            new_state = (
                np.einsum("nij,nj->ni", data["A"], state)
                + data["B"] * applied[:, None] + disturbance[:, :, tick]
            )
            state[active] = new_state[active]
            errors[active] += state[active, 0] ** 2
            if mode == "tail" and tick >= 40:
                tail_errors[active] += state[active, 0] ** 2
            lengths[active] += 1
            active &= (abs(state[:, 0]) <= 2.5) & (abs(state[:, 2]) <= .5)
        rmse = np.sqrt((errors + (1000-lengths)*np.maximum(2.5, abs(state[:, 0]))**2)/1001)
        scores = condition_weights @ rmse.reshape(len(seeds), -1).mean(axis=0)
        tail_scores = condition_weights @ np.sqrt(tail_errors / 960).reshape(len(seeds), -1).mean(axis=0)
        return scores, int(active.sum()), samples, tail_scores

    original, completed, samples, original_tail = rollout(weights, collect=True)
    feature_rows, observation_rows = zip(*samples)
    feature_rows = np.stack(feature_rows)
    observation_rows = np.stack(observation_rows)
    directions = []
    for group in groups:
        x = feature_rows[:, group].reshape(-1, 65).astype(float)
        obs = observation_rows[:, group].reshape(-1, 11).astype(float)
        inputs = obs[:, [0, 1, 2, 3, 5, 6, 7]]
        scales = np.maximum(np.sqrt(np.mean(inputs**2, axis=0)), 1e-5)
        # Seek stronger steering adjustments where recent observed latency is low.
        # The fitted correction is folded into the existing linear output head.
        attenuation = np.exp(-10.0 * np.maximum(obs[:, 9], obs[:, 10]))
        target = inputs / scales * attenuation[:, None] * .004
        if mode == "nominal":
            target *= 8
        ridge = 0.00003 if mode == "nominal" else 0.003
        if mode == "mixed":
            recent = np.maximum(obs[:, 9], obs[:, 10])
            transient = (recent > .002) & (recent < .7)
            target = np.column_stack([
                target,
                inputs[:, :4] / scales[:4] * transient[:, None] * .005,
                obs[:, 9:11] * transient[:, None] * .01,
            ])
            ridge = .00025
        if mode == "transient":
            recent = np.maximum(obs[:, 9], obs[:, 10])
            large_error = np.clip((np.abs(obs[:, 0]) - .08) / .32, 0, 1)
            weight = np.exp(-30 * recent) * large_error
            target = np.column_stack([
                target,
                inputs[:, :4] / scales[:4] * weight[:, None] * .035,
            ])
            ridge = .0001
        if mode in {"initial", "recovery"}:
            recent = np.maximum(obs[:, 9], obs[:, 10])
            response = np.exp(-25 * recent) * np.tanh(-8 * obs[:, 0])
            initial_rows = np.arange(len(x)) < count
            sample_weights = np.where(initial_rows, 10., 1.)
            target = np.c_[.45 * response, .25 * response * np.clip(abs(obs[:, 0]), 0, 1)]
            if mode == "recovery":
                delayed = np.clip((recent - .05) / .45, 0, 1)
                target = np.column_stack([
                    target,
                    inputs[:, :4] / scales[:4] * delayed[:, None] * .015,
                    inputs[:, 4:6] / scales[4:6] * delayed[:, None] * .004,
                ])
            x = x * np.sqrt(sample_weights)[:, None]
            target = target * np.sqrt(sample_weights)[:, None]
            ridge = .00003
        if mode == "tail":
            recent = np.maximum(obs[:, 9], obs[:, 10])
            quiet = np.exp(-20 * recent) * np.exp(-4 * np.abs(obs[:, 0]))
            target = inputs / scales * quiet[:, None] * .002
            ridge = .00005
        penalty = ridge * len(x) * np.eye(x.shape[1])
        directions.append(np.linalg.solve(x.T @ x + penalty, x.T @ target))
    directions = np.stack(directions)
    if mode in {"blend", "robust"}:
        blend_directions = []
        for i, group in enumerate(groups):
            x = feature_rows[:, group].reshape(-1, 65).astype(float)
            obs = observation_rows[:, group].reshape(-1, 11)
            other_heads = []
            for name in (("broad_trial", "blend_trial") if mode == "robust" else ("trial", "nominal_trial")):
                other = PPO.load(HERE / name / f"policy_{seeds[i]}.zip", device="cpu").policy.action_net
                other_heads.append(np.r_[other.weight.detach().numpy().ravel(), other.bias.detach().numpy()])
            differences = np.stack(other_heads) - weights[i]
            high_delay = (obs[:, 9] > .8).astype(float)
            gated_target = (x @ differences.T) * np.c_[high_delay, 1 - high_delay]
            ridge = .00001 * len(x) * np.eye(x.shape[1])
            gated = np.linalg.solve(x.T @ x + ridge, x.T @ gated_target)
            blend_directions.append(np.column_stack([gated, differences.T, directions[i]]))
        directions = np.stack(blend_directions)
    validation = STUDY / "results" / "validation"
    references = [
        read_means(sorted(validation.glob("limited_ppo_*.csv"))),
        read_means([validation / "fast_lqr_-1.csv"]),
        read_means([validation_mpc]),
    ]
    if mode in {"broad", "blend", "robust", "mixed", "transient", "initial", "tail", "recovery"}:
        references = [
            read_means([validation_mpc.with_name(f"{name}.csv")])
            for name in ("limited_ppo", "fast_lqr", "matched_mpc")
        ]
    target = np.array([min(scores[key] for scores in references) for key in keys])
    original_wins = original < target
    trials = []
    best = None

    def objective(parameters):
        nonlocal best
        heads = (weights + directions @ parameters).astype(np.float32)
        scores, count_finished, _, tail_scores = rollout(heads)
        ratio = scores / target
        # Balance the geometric improvement with emphasis on close races.
        loss = float(
            np.mean(np.log(ratio))
            + .35 * np.mean(np.tanh(4 * np.log(ratio)))
            + 10 * (len(cases) - count_finished)
            + 4 * np.sum(np.maximum(ratio[original_wins] - 1, 0))
        )
        if mode == "nominal":
            loss += float(2 * np.mean(np.log(ratio[nominal])))
        if mode in {"broad", "blend", "robust", "mixed", "transient", "initial", "tail", "recovery"}:
            loss += float(np.mean(np.maximum(ratio[~nominal] - .88, 0)))
        if mode == "robust":
            delayed = ratio[~nominal]
            loss += float(.8 * np.max(delayed) + .7 * np.mean(delayed))
        if mode == "mixed":
            nominal_delta = scores[nominal] / original[nominal] - 1
            ratio_mixed = ratio[mixed]
            loss = float(
                np.max(ratio_mixed) + np.mean(ratio_mixed)
                + 6 * np.sum(np.maximum(ratio[fixed] - .985, 0))
                + 3 * np.sum(np.maximum(nominal_delta - .01, 0))
                + 10 * (len(cases) - count_finished)
            )
        if mode in {"transient", "initial", "recovery"}:
            loss = float(
                np.mean(np.log(ratio[nominal]))
                + 6 * np.sum(np.maximum(ratio[~nominal] - .99, 0))
                + 10 * (len(cases) - count_finished)
            )
        if mode == "tail":
            tail_ratio = (tail_scores[nominal] + 1e-9) / (original_tail[nominal] + 1e-9)
            loss = float(
                np.mean(np.log(tail_ratio))
                + 7 * np.sum(np.maximum(ratio[~nominal] - .99, 0))
                + 4 * np.sum(np.maximum(scores[nominal] / original[nominal] - 1., 0))
                + 10 * (len(cases) - count_finished)
            )
        record = {
            "parameters": parameters.tolist(), "mean_rmse": scores.tolist(),
            "validation_wins": int(np.sum(ratio < 1)),
            "completion": count_finished, "loss": loss,
        }
        if mode == "tail":
            record["tail_rmse"] = tail_scores.tolist()
        trials.append(record)
        # Retain existing validation wins and require complete episodes.
        if count_finished == len(cases) and np.all(ratio[original_wins] < 1):
            preference = (-record["validation_wins"], loss)
            if best is None or preference < best[0]:
                best = preference, record, heads.copy()
                print("Validation candidate", record["validation_wins"], loss, flush=True)
        return loss

    objective(np.zeros(directions.shape[-1]))
    start = np.zeros(directions.shape[-1])
    if mode in {"blend", "robust"}:
        for indices in ([0, 1], [0, 3], [2, 3], [2], [3]):
            trial = np.zeros(directions.shape[-1])
            trial[indices] = 1
            objective(trial)
        start = np.asarray(best[1]["parameters"])
    bound = .8 if mode in {"mixed", "transient", "tail"} else 3
    if mode == "initial":
        for alpha in (.025, .05, .1, .2, .4, .6, 1., 1.5, 2.):
            objective(np.array([alpha, 0.]))
            objective(np.array([0., alpha]))
        start = np.asarray(best[1]["parameters"])
    if mode == "recovery":
        for alpha in (.01, .025, .05, .1):
            trial = start.copy()
            trial[0] = alpha
            objective(trial)
        start = np.asarray(best[1]["parameters"])
        bound = .4
    minimize(
        objective, start, method="Powell",
        bounds=[(-bound, bound)] * directions.shape[-1],
        options={"maxfev": {"nominal": 380, "robust": 320, "mixed": 175, "transient": 175, "initial": 65, "tail": 110, "recovery": 135}.get(mode, 180), "ftol": .0003},
    )
    _, selected, heads = best
    for seed, model, head in zip(seeds, models, heads):
        with torch.no_grad():
            model.policy.action_net.weight.copy_(torch.from_numpy(head[:-1][None, :]))
            model.policy.action_net.bias.copy_(torch.from_numpy(head[-1:]))
        model.save(destination / f"policy_{seed}.zip")
    (destination / "validation.json").write_text(json.dumps({
        "mode": mode, "conditions": keys, "benchmark_rmse": target.tolist(), "baseline": trials[0],
        "selected": selected, "trials": trials,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
