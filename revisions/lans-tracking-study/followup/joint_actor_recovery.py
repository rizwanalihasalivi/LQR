"""Consolidate nominal and delayed teachers into the existing actor."""
import argparse
import json

import numpy as np
from stable_baselines3 import PPO
import torch

from distill import HERE, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases

TRAIN_SEEDS = [22553, 22554, 22555, 22556, 22557]
SCALES = [(.90, 1.07), (1.12, .95)]
CHECKPOINTS = (500, 1500, 4000)


def collect(prior, nominal_teacher, student=None, observed_teacher=False):
    observations, labels, regimes, reset_rows = [], [], [], []
    seeds = TRAIN_SEEDS if student is None else [22564, 22565, 22566, 22567, 22568]
    scales = SCALES if student is None else [(.92, .94), (1.14, 1.08)]
    if observed_teacher:
        seeds = [22575, 22576, 22577, 22578, 22579]
        scales = [(.89, 1.09), (1.14, .97)]
    for case in build_cases(seeds, scales):
        env = make_env(case)
        regime = ("none", "mixed", "fixed200").index(case["delay"])
        # Get the retained and more aggressive teacher's state distributions.
        pulses = (0., .5, 1.) if observed_teacher else (0., .15, .5)
        for pulse in pulses:
            obs, _ = env.reset(seed=case["seed"])
            for tick in range(1000):
                reference = np.clip(prior(obs), -1., 1.)
                teacher = np.clip(nominal_teacher(obs), -1., 1.)
                if tick < 65 or tick % 10 == 0:
                    observations.append(obs)
                    if observed_teacher:
                        mix = np.clip(max(obs[9], obs[10]) / .03, 0., 1.)
                        labels.append(teacher + mix * (reference - teacher))
                    else:
                        labels.append(teacher if regime == 0 else reference)
                    regimes.append(regime)
                    reset_rows.append(tick == 0)
                if regime == 0:
                    action = reference if pulse == 0 else teacher
                else:
                    action = reference if tick > 0 else reference + pulse * (teacher - reference)
                if student is not None and pulse == .5:
                    action = np.clip(student(obs), -1., 1.)
                obs, _, terminated, truncated, _ = env.step([action])
                if terminated or truncated:
                    break
    return seeds, scales, (
        torch.tensor(np.stack(observations)),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(regimes),
        torch.tensor(reset_rows),
    )


def train(seed, aggregate_student=False, observed_teacher=False):
    torch.set_num_threads(1)
    retained = HERE / "guarded_round2" / "blend_.1" / f"policy_{seed}.zip"
    nominal = HERE / "nominal_oracle_actor" / f"policy_{seed}_2400.zip"
    baseline = PPO.load(retained, device="cpu")
    start = HERE / "joint_actor_recovery" / f"policy_{seed}_4000.zip" if aggregate_student else nominal
    model = PPO.load(start, device="cpu")
    teacher = PPO.load(nominal, device="cpu") if aggregate_student else model
    student = actor(model) if aggregate_student else None
    training_seeds, scales, samples = collect(actor(baseline), actor(teacher), student, observed_teacher)
    observations, labels, regimes, reset_rows = samples
    # Delayed reset states also occur in nominal episodes. Do not train
    # contradictory delayed labels at those observations.
    regime_samples = [
        torch.where((regimes == i) & (~reset_rows if i and not observed_teacher else torch.ones_like(reset_rows)))[0]
        for i in range(3)
    ]
    hidden = model.policy.mlp_extractor.policy_net
    head = model.policy.action_net
    optimizer = torch.optim.Adam([*hidden.parameters(), *head.parameters()], lr=.00015 if aggregate_student else .00025)
    generator = torch.Generator().manual_seed(seed)
    output = HERE / ("joint_actor_aggregate" if aggregate_student else "joint_actor_recovery")
    if observed_teacher:
        output = HERE / "joint_actor_observed_teacher"
    output.mkdir(exist_ok=True)
    history = []
    checkpoints = (500, 1500, 3000) if aggregate_student else CHECKPOINTS
    if observed_teacher:
        checkpoints = (500, 2000, 5000)
    for step in range(1, checkpoints[-1] + 1):
        index = torch.cat([
            pool[torch.randint(len(pool), (256,), generator=generator)]
            for pool in regime_samples
        ])
        prediction = head(hidden(observations[index])).squeeze(-1)
        group_loss = torch.nn.functional.smooth_l1_loss(
            prediction, labels[index], beta=.015, reduction="none"
        ).reshape(3, -1).mean(dim=1)
        loss = group_loss.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            model.save(output / f"policy_{seed}_{step}.zip")
            values = group_loss.detach().tolist()
            history.append({"update": step, "group_loss": values})
            print(seed, step, values, flush=True)
    (output / f"training_{seed}.json").write_text(json.dumps({
        "source": str(start.relative_to(HERE)),
        "nominal_teacher": str(nominal.relative_to(HERE)),
        "delayed_teacher": str(retained.relative_to(HERE)),
        "training_seeds": training_seeds,
        "teacher_uses_observed_delay": observed_teacher,
        "scales": scales,
        "samples_by_regime": [len(pool) for pool in regime_samples],
        "history": history,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=POLICY_SEEDS)
    parser.add_argument("--aggregate-student", action="store_true")
    parser.add_argument("--observed-teacher", action="store_true")
    args = parser.parse_args()
    train(args.seed, args.aggregate_student, args.observed_teacher)
