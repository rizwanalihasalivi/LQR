"""Measure shared reset observations and the effect of a nominal-first-command intervention."""
import csv
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO

from distill import HERE, POLICY_SEEDS, actor, make_env
from batch_sim import build_cases
from calibration import SCALES, SEEDS
from receding_nominal_targets import SteeringTeacher


def replay(case, predict, first=None):
    env = make_env(case)
    obs, _ = env.reset(seed=case["seed"])
    squared_error = float(env.x[0])**2
    for tick in range(1000):
        action = first if tick == 0 and first is not None else predict(obs)
        obs, _, terminated, truncated, info = env.step([action])
        squared_error += info["error"]**2
        if terminated or truncated:
            break
    squared_error += (999 - tick) * max(2.5, abs(info["error"]))**2
    return float(np.sqrt(squared_error / 1001)), bool(truncated and not terminated)


def main():
    paths = [HERE / "anchor_prefix_focus" / f"policy_{seed}_150.zip" for seed in POLICY_SEEDS]
    actors = [actor(PPO.load(path, device="cpu")) for path in paths]
    results = defaultdict(list)
    for case in build_cases(SEEDS, SCALES):
        if case["delay"] != "fixed200":
            continue
        fixed = make_env(case)
        delayed_obs, _ = fixed.reset(seed=case["seed"])
        nominal = make_env({**case, "delay": "none"})
        nominal_obs, _ = nominal.reset(seed=case["seed"])
        teacher = SteeringTeacher(nominal)
        raw = teacher.action(nominal, 0) + float(nominal.K @ nominal.x)
        for predict in actors:
            baseline = replay(case, predict)
            intervention = replay(case, predict, raw)
            results[case["path"], case["amplitude"], case["time_scale"]].append(
                (np.array_equal(delayed_obs, nominal_obs), *baseline, *intervention)
            )

    with (HERE / "reset_delay_ambiguity.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "amplitude", "time_scale", "episodes", "same_reset_obs",
                         "baseline_rmse", "baseline_completed", "teacher_first_rmse",
                         "teacher_first_completed"))
        for key, records in sorted(results.items()):
            same, baseline, completed, intervention, intervention_completed = np.array(records).T
            writer.writerow((*key, len(records), int(same.sum()), baseline.mean(), int(completed.sum()),
                             intervention.mean(), int(intervention_completed.sum())))


if __name__ == "__main__":
    main()
