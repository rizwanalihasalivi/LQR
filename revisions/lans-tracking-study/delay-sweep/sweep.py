import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

import torch

# The tracking study and its sibling both have a module named protocol.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from evaluate import load, NetworkEnv, MatchedMPC, P
from dynamic_env import HORIZON

DELAYS = (0, 50, 100, 150, 200, 300, 400, 700, 1000, 1500, 2000)
CONTROLLERS = (
    ("LQR", "lqr", None, (-1,)),
    ("PPO", "ppo", "limited_ppo", (72, 73, 74)),
    ("MPC", "mpc", None, (-1,)),
    ("LANS", "lans", "limited_lans", (72, 73, 74)),
)
POLICIES = {}
OUT = Path(__file__).resolve().parent

def run(job):
    delay, label, variant, checkpoint, train_seed = job
    policy = POLICIES.get((checkpoint, train_seed))
    rows = []
    for amplitude, time_scale in P["test"]["scales"]:
        for path in P["test"]["families"]:
            for physical_seed in P["test"]["seeds"]:
                env = NetworkEnv(kind=variant, path=path, mass=1650.,
                                 delay_mode=f"fixed{delay}", amplitude=amplitude,
                                 time_scale=time_scale)
                assert env.delay_scope == "full"
                obs, _ = env.reset(seed=physical_seed)
                sse = float(env.x[0]) ** 2
                mpc = MatchedMPC() if label == "MPC" else None
                for steps in range(1, HORIZON + 1):
                    if policy is not None:
                        action = policy.predict(obs, deterministic=True)[0]
                    elif mpc is not None:
                        action = [mpc.action(env) / .5]
                    else:
                        action = [0.]
                    obs, _, terminated, truncated, info = env.step(action)
                    assert info["delay_ms"] == delay
                    error = info["error"]
                    sse += error ** 2
                    if terminated or truncated:
                        break
                rows.append(dict(delay_ms=delay, controller=label, train_seed=train_seed,
                                 path=path, physical_seed=physical_seed, amplitude=amplitude,
                                 time_scale=time_scale, steps=steps,
                                 completed=int(truncated and not terminated),
                                 rmse_observed_m=math.sqrt(sse / (steps + 1)),
                                 rmse_penalized_m=math.sqrt((sse + (HORIZON - steps) *
                                                            max(2.5, abs(error)) ** 2) /
                                                           (HORIZON + 1))))
    return rows

def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for _, _, checkpoint, seeds in CONTROLLERS:
        if checkpoint:
            for seed in seeds:
                POLICIES[checkpoint, seed] = load(checkpoint, seed)
    jobs = [(delay, label, variant, checkpoint, seed) for delay in DELAYS
            for label, variant, checkpoint, seeds in CONTROLLERS for seed in seeds]
    episodes = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for index, batch in enumerate(pool.map(run, jobs), start=1):
            episodes.extend(batch)
            if index % 8 == 0:
                print(f"Finished delay {batch[0]['delay_ms']} ms ({index}/{len(jobs)})", flush=True)
    assert len(episodes) == 3520
    write_csv("episodes.csv", episodes)
    summary = []
    for delay in DELAYS:
        group = []
        for label, _, _, seeds in CONTROLLERS:
            rows = [r for r in episodes if r["delay_ms"] == delay and r["controller"] == label]
            assert len(rows) == 40 * len(seeds)
            group.append(dict(delay_ms=delay, controller=label,
                              rmse_penalized_m=mean(r["rmse_penalized_m"] for r in rows),
                              rmse_observed_m=mean(r["rmse_observed_m"] for r in rows),
                              completed=sum(r["completed"] for r in rows), episodes=len(rows)))
        for rank, row in enumerate(sorted(group, key=lambda r: r["rmse_penalized_m"]), 1):
            summary.append(dict(rank=rank, **row))
    write_csv("summary.csv", summary)
