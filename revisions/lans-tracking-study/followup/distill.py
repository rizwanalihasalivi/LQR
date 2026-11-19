"""Offline steering distillation with the existing LANS controller at deployment."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_discrete_are
from stable_baselines3 import PPO
import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "code"))
from features import NetworkEnv
from batch_sim import build_cases

POLICY_SEEDS = (72, 73, 74)


def offline_expert(env):
    """Use reference access and plant mass solely for offline training labels."""
    a, b = env.Ad, env.Bd
    q = np.diag([1000, 0., .02, 0.])
    r = np.array([[.0002]])
    p = solve_discrete_are(a, b[:, None], q, r)
    denominator = float((r + b @ p @ b).item())
    feedback = (b @ p @ a) / denominator
    closed_loop = a - np.outer(b, feedback)
    disturbance = (env.Gd[:, None] * env.ref[2] + env.Hd[:, None] * env.ref[3]) / env.vx
    future = np.zeros(4)
    feedforward = np.empty(disturbance.shape[1])
    for tick in reversed(range(len(feedforward))):
        term = p @ disturbance[:, tick] + future
        feedforward[tick] = (b @ term) / denominator
        future = closed_loop.T @ term
    return feedback, feedforward


def make_env(case):
    config = {k: v for k, v in case.items() if k not in {"delay", "seed"}}
    return NetworkEnv(kind="lans", delay_mode=case["delay"], **config)


def actor(model):
    net = model.policy.mlp_extractor.policy_net
    head = model.policy.action_net
    w1, b1, w2, b2 = [p.detach().numpy().copy() for p in net.parameters()]
    w3, b3 = [p.detach().numpy().copy() for p in head.parameters()]
    def predict(obs):
        return float((w3 @ np.maximum(w2 @ np.maximum(w1 @ obs + b1, 0) + b2, 0) + b3)[0])
    return predict


def dataset(model, constrained=False, joint=False, guarded=False, second_pass=False, nominal_focus=False, mixed_focus=False, ppo_teacher=None, strong_ppo=False):
    if constrained:
        from constrained_expert import plan
    if joint:
        from fixed_delay_ceil import fixed_commands
        from core import beta
    seeds = [22503, 22504, 22505] if second_pass else [22500, 22501, 22502]
    scales = [(.97, 1.11), (1.05, .92)] if second_pass else [(.93, .89), (1.09, 1.08)]
    if nominal_focus:
        seeds = [22506, 22507, 22508]
        scales = [(.91, .95), (1.13, 1.06)]
    if mixed_focus:
        from mixed_expert import commands
        seeds = [22509, 22510, 22511]
        scales = [(.86, 1.01), (1.16, 1.04)]
    if ppo_teacher is not None:
        seeds = [22512, 22513, 22514]
        scales = [(.89, 1.04), (1.11, .93)]
    if strong_ppo:
        seeds = [22515, 22516, 22517]
        scales = [(.90, 1.09), (1.15, .95)]
    nominal_fraction = .05 if nominal_focus else .025
    cases = build_cases(seeds, scales=scales)
    predict = actor(model)
    rng = np.random.default_rng(7642)
    samples, labels, nominal = [], [], []
    for case in cases:
        env = make_env(case)
        obs, _ = env.reset(seed=case["seed"])
        no_delay = case["delay"] == "none"
        fixed_delay = joint and not (nominal_focus or mixed_focus) and case["delay"] == "fixed200"
        delayed_commands = fixed_commands(case) if fixed_delay else None
        use_mixed_expert = mixed_focus and case["delay"] == "mixed"
        mixed_commands = commands(env)[0] if use_mixed_expert and ppo_teacher is None else None
        feedback, feedforward = offline_expert(env) if no_delay and not mixed_focus else (None, None)
        if constrained and no_delay and not mixed_focus:
            optimal = plan(env, .00005, .0001)
            reference_states = [env.x.copy()]
            for tick, steering in enumerate(optimal):
                state = (
                    env.Ad @ reference_states[-1] + env.Bd * steering
                    + (env.Gd * env.ref[2, tick] + env.Hd * env.ref[3, tick]) / env.vx
                )
                reference_states.append(state)
        for tick in range(1000):
            current = predict(obs)
            if use_mixed_expert:
                if ppo_teacher is None:
                    command = mixed_commands[tick]
                    fraction = .04
                else:
                    command = .5 * np.clip(ppo_teacher(obs), -1, 1)
                    fraction = .65 if strong_ppo else .12
                expert = np.clip((command + env.K @ env.x) / beta(env.delays[tick]), -1, 1)
                target = current + fraction * (expert - current)
                action = current
            elif no_delay and not mixed_focus:
                raw = -feedback @ env.x - feedforward[tick]
                if constrained:
                    raw = optimal[tick] - feedback @ (env.x - reference_states[tick])
                steering = np.clip(raw, -.5, .5)
                target = np.clip(steering + env.K @ env.x, -1, 1)
                # Mix expert and reference trajectories for state coverage.
                action = target if (case["seed"] != 22500 or tick >= 125) else current
                if guarded:
                    target = current + nominal_fraction * (target - current)
                    action = current
            elif fixed_delay and tick < len(delayed_commands):
                target = action = np.clip((delayed_commands[tick] + env.K @ env.x) / beta(200), -1, 1)
                if guarded:
                    target = current + .01 * (target - current)
                    action = current
            else:
                target = action = current
            if tick < 120 or tick % 8 == 0:
                samples.append(obs)
                labels.append(target)
                nominal.append(no_delay)
                if no_delay and not mixed_focus:
                    augmented = obs.copy()
                    perturbation = rng.normal(0, [.003, .010, .001, .004])
                    augmented[:4] += perturbation / [.1, .5, .05, .2]
                    state = env.x + perturbation
                    steering = np.clip(raw - feedback @ perturbation, -.5, .5)
                    samples.append(augmented)
                    expert = np.clip(steering + env.K @ state, -1, 1)
                    if guarded:
                        augmented_current = predict(augmented)
                        expert = augmented_current + nominal_fraction * (expert - augmented_current)
                    labels.append(expert)
                    nominal.append(True)
            obs, _, term, trunc, _ = env.step([action])
            if term or trunc:
                break
    return torch.tensor(np.stack(samples)), torch.tensor(labels, dtype=torch.float32), torch.tensor(nominal)


def main():
    torch.set_num_threads(1)
    seed = int(sys.argv[1])
    assert seed in POLICY_SEEDS
    second_pass = "--guard-round-2" in sys.argv
    nominal_focus = "--nominal-focus" in sys.argv
    strong_ppo = "--ppo-mixed-strong" in sys.argv
    ppo_mixed = strong_ppo or "--ppo-mixed" in sys.argv
    mixed_focus = ppo_mixed or "--mixed-focus" in sys.argv
    guarded = mixed_focus or nominal_focus or second_pass or "--joint-guarded" in sys.argv
    joint = guarded or "--joint-expert" in sys.argv
    constrained = joint or "--constrained" in sys.argv
    source = "initial_trial" if joint else ("broad_trial" if constrained else "trial")
    model_path = (
        HERE / "guarded_distilled" / f"policy_{seed}_600.zip" if second_pass
        else HERE / source / f"policy_{seed}.zip"
    )
    if nominal_focus or mixed_focus:
        model_path = HERE / "guarded_round2" / "blend_.1" / f"policy_{seed}.zip"
    model = PPO.load(model_path, device="cpu")
    teacher = None
    if ppo_mixed:
        frozen = STUDY / "models" / f"limited_ppo_{seed}" / "policy.zip"
        teacher = actor(PPO.load(frozen, device="cpu"))
    obs, target, nominal = dataset(
        model, constrained, joint, guarded, second_pass, nominal_focus, mixed_focus, teacher, strong_ppo
    )
    hidden = model.policy.mlp_extractor.policy_net
    head = model.policy.action_net
    optimizer = torch.optim.Adam([*hidden.parameters(), *head.parameters()], lr=.000025 if guarded else .00025)
    random = torch.Generator().manual_seed(seed)
    folder = "guarded_distilled" if guarded else "joint_distilled"
    out = HERE / (folder if joint else ("constrained_distilled" if constrained else "distilled"))
    if second_pass:
        out = HERE / "guarded_round2"
    if nominal_focus:
        out = HERE / "nominal_focused"
    if mixed_focus:
        out = HERE / "mixed_focused"
    if ppo_mixed:
        out = HERE / ("ppo_mixed_strong" if strong_ppo else "ppo_mixed")
    out.mkdir(exist_ok=True)
    checkpoints = (200, 600, 1200) if second_pass else (50, 200, 600) if guarded else (100, 400, 1000) if joint else ((400, 1200, 3200) if constrained else (200, 600, 1800))
    if nominal_focus or mixed_focus:
        checkpoints = (100, 400, 1000)
    for step in range(checkpoints[-1]):
        idx = torch.randint(len(obs), (512,), generator=random)
        prediction = head(hidden(obs[idx])).squeeze(-1)
        weights = torch.where(nominal[idx], .5 if guarded else (.025 if constrained else .06), 1.)
        if nominal_focus:
            weights = torch.where(nominal[idx], .2, 1.)
        if mixed_focus:
            weights = torch.ones_like(weights)
        loss = ((prediction - target[idx]).square() * weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step + 1 in checkpoints:
            model.save(out / f"policy_{seed}_{step + 1}.zip")
            print(seed, step + 1, float(loss.detach()), flush=True)
    (out / f"training_{seed}.json").write_text(json.dumps({
        "source": str(model_path.relative_to(HERE)),
        "training_cases": [22515, 22516, 22517] if strong_ppo else [22512, 22513, 22514] if ppo_mixed else [22509, 22510, 22511] if mixed_focus else [22506, 22507, 22508] if nominal_focus else ([22503, 22504, 22505] if second_pass else [22500, 22501, 22502]),
        "samples": len(obs),
        "nominal_samples": int(nominal.sum()),
        "updates": checkpoints[-1],
        "fixed_delay_expert": joint and not (nominal_focus or mixed_focus),
        "mixed_delay_expert": mixed_focus and not ppo_mixed,
        "frozen_ppo_teacher": ppo_mixed,
        "ppo_label_fraction": (.65 if strong_ppo else .12) if ppo_mixed else None,
        "guarded_labels": guarded,
        "scope": "Offline supervised revision of the existing PPO-trained actor.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
