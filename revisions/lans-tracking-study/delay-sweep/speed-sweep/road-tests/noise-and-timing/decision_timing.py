"""Measure warmed, single-sample CPU steering duration on common replay states."""
import argparse
from itertools import product
from pathlib import Path
import json
import os
import platform
from time import perf_counter_ns

import numpy as np
import scipy
import osqp
import stable_baselines3
import torch

from noise_replay import HERE, DELAYS, ROADS, SPEEDS, FRESH_SEEDS, configure, load_policies, sweep, write


def decide(env, model, mpc, delay):
    if model is not None:
        action = model.predict(env.obs(), deterministic=True)[0]
    elif mpc is not None:
        action = (mpc.action(env) / .5,)
    else:
        action = (0.,)
    return env.compose(action, delay)[0]


def scenario(speed, delay, path):
    policy_seed = sweep.TRAIN_SEEDS[0]
    seed = FRESH_SEEDS[0]
    amplitude, scale = sweep.sweep.P["test"]["scales"][0]
    envs = {}
    for controller in sweep.CONTROLLERS:
        env, horizon, ref = configure(controller, path, delay, speed, amplitude, scale, "fixed_road")
        env.reset(seed=seed)
        env.ref = ref
        env.delays = np.full(horizon + 1, delay, dtype=float)
        envs[controller] = env
    mpc = sweep.SweepMPC(mass=env.mass, vx=env.vx)
    calls = 0
    solve = mpc.solver.solve

    def count_solve(*args, **kwargs):
        nonlocal calls
        calls += 1
        return solve(*args, **kwargs)

    mpc.solver.solve = count_solve
    reference = envs["LANS"]
    reference_policy = sweep.POLICIES["LANS", policy_seed]
    participants = [
        (controller, envs[controller], sweep.POLICIES.get((controller, policy_seed)),
         mpc if controller == "MPC" else None)
        for controller in sweep.CONTROLLERS
    ]
    # Warm the Python/Torch paths before taking measurements. The MPC warm start
    # used for timing starts again from the scenario's normal zero initialization.
    for _ in range(40):
        for _, env, model, optimizer in participants:
            decide(env, model, optimizer, delay)
    mpc.last.fill(0.)
    mpc.solver.warm_start(x=np.zeros(mpc.N), y=np.zeros(mpc.N))
    mpc.failures = 0
    for tick in range(horizon):
        offset = tick % len(participants)
        for controller, env, model, optimizer in participants[offset:] + participants[:offset]:
            env.x = reference.x
            env.k = reference.k
            env.prev_u = reference.prev_u
            env.last_delay = reference.last_delay
            if env is not reference:
                env.history = reference.history
            before = calls
            failures = mpc.failures
            start = perf_counter_ns()
            decide(env, model, optimizer, delay)
            elapsed = perf_counter_ns() - start
            yield dict(speed_kmh=speed, delay_ms=delay, path=path, tick=tick, controller=controller,
                       decision_ns=elapsed, qp_solved=int(calls > before),
                       fallback=int(mpc.failures > failures))
        action = reference_policy.predict(reference.obs(), deterministic=True)[0]
        _, _, failed, _, _ = reference.step(action)
        if failed:
            break


def stats(label, times):
    us = np.asarray(times) / 1000.
    return dict(controller=label, decisions=len(us), mean_us=float(us.mean()),
                median_us=float(np.median(us)), p95_us=float(np.quantile(us, .95)),
                p99_us=float(np.quantile(us, .99)), max_us=float(max(us)),
                over_10ms=int(sum(us > 10000.)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    load_policies()
    rows = []
    for repeat in range(1, args.repeats + 1):
        rows.extend(dict(repeat=repeat, **row) for cell in product(SPEEDS, DELAYS, ROADS)
                    for row in scenario(*cell))
        print(f"Timing repeat {repeat}/{args.repeats} finished", flush=True)
    times = {
        controller: [r["decision_ns"] for r in rows if r["controller"] == controller]
        for controller in sweep.CONTROLLERS
    }
    assert len({len(x) for x in times.values()}) == 1
    summary = [stats(controller, values) for controller, values in times.items()]
    for constrained in (0, 1):
        values = [r["decision_ns"] for r in rows if r["controller"] == "MPC" and r["qp_solved"] == constrained]
        if values:
            summary.append(stats("MPC QP" if constrained else "MPC analytic", values))
    write("timing_decisions.csv", rows)
    write("timing_summary.csv", summary)
    write("timing_by_repeat.csv", [
        dict(repeat=repeat, **stats(controller, [r["decision_ns"] for r in rows
                                                if r["repeat"] == repeat and r["controller"] == controller]))
        for repeat, controller in product(range(1, args.repeats + 1), sweep.CONTROLLERS)
    ])
    cpuinfo = Path("/proc/cpuinfo").read_text()
    model = next(line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name"))
    metadata = dict(cpu=model, logical_cpu=cpu, python=platform.python_version(), machine=platform.machine(),
                    numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                    stable_baselines3=stable_baselines3.__version__, torch_threads=torch.get_num_threads(),
                    osqp=osqp.__version__)
    (HERE / "timing_host.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
