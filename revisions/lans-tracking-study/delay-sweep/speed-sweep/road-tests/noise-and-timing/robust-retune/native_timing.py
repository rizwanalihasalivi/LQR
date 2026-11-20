"""Benchmark shared native actor inference and LQR against the existing MPC."""
import argparse
from collections import defaultdict
import ctypes
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
from stable_baselines3 import PPO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import decision_timing as timing
from noise_replay import DELAYS, ROADS, SPEEDS, load_policies, sweep, write
from features import T

FLOAT = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
DOUBLE = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


class NativeControllers:
    def __init__(self, library):
        self.library = ctypes.CDLL(library)
        self.blas = ctypes.CDLL(str(Path(torch.__file__).parent / "lib" / "libtorch_cpu.so"))
        self.gemm = ctypes.cast(self.blas.sgemm_, ctypes.c_void_p)
        self.library.create_actor.argtypes = [FLOAT] * 6 + [ctypes.c_void_p]
        self.library.create_actor.restype = ctypes.c_void_p
        self.library.destroy_actor.argtypes = [ctypes.c_void_p]
        self.library.destroy_actor.restype = None
        self.library.steer_lqr.argtypes = [DOUBLE, DOUBLE]
        self.library.steer_lqr.restype = ctypes.c_double
        self.library.steer_actor.argtypes = ([ctypes.c_void_p] + [DOUBLE] * 4 + [ctypes.c_int] * 2
                                              + [ctypes.c_double] * 6 + [ctypes.c_int])
        self.library.steer_actor.restype = ctypes.c_double
        self.actors = {}
        self.transform = np.ascontiguousarray(T)

    def register(self, model):
        first, _, second, _ = model.policy.mlp_extractor.policy_net
        output = model.policy.action_net
        arrays = [layer.detach().numpy() for net in (first, second, output)
                  for layer in (net.weight, net.bias)]
        self.actors[model] = self.library.create_actor(*arrays, self.gemm)

    def decide(self, env, model, mpc, delay):
        if mpc is not None:
            return env.compose((mpc.action(env) / .5,), delay)[0]
        if model is None:
            return self.library.steer_lqr(env.x, env.K)
        return self.library.steer_actor(
            self.actors[model], env.x, env.K, env.ref, self.transform, env.ref.shape[1], env.k, env.vx,
            env.last_delay, sum(env.history) / len(env.history) if env.history else 0.,
            env.residual_scale, env.beta_decay, delay, int(env.variant == "lans"),
        )

    def close(self):
        for actor in self.actors.values():
            self.library.destroy_actor(actor)


def run_benchmark(native, repeats, compiler, variant):
    original = timing.decide
    deviation = defaultdict(float)
    samples = defaultdict(int)

    def compare(env, model, mpc, delay):
        output = native.decide(env, model, mpc, delay)
        if mpc is None:
            deviation[env.variant] = max(deviation[env.variant], abs(output - original(env, model, mpc, delay)))
            samples[env.variant] += 1
        return output

    tape = tuple(product(SPEEDS, DELAYS, ROADS))
    timing.decide = compare
    for cell in tape:
        for _ in timing.scenario(*cell):
            pass
    if not all(np.isfinite(x) and x < 5e-6 for x in deviation.values()):
        raise RuntimeError(f"Native steering failed the parity check: {dict(deviation)}")
    print("Maximum steering difference in radians:", dict(deviation), flush=True)
    timing.decide = original
    eager_times = defaultdict(list)
    for cell in tape:
        for row in timing.scenario(*cell):
            eager_times[row["controller"]].append(row["decision_ns"])
    write(HERE / "matched_eager_timing_summary.csv",
          [timing.stats(label, values) for label, values in eager_times.items()])
    print("Matched eager timing finished", flush=True)
    timing.decide = native.decide

    rows = []
    for repeat in range(1, repeats + 1):
        rows.extend(dict(repeat=repeat, **row) for cell in tape for row in timing.scenario(*cell))
        print(f"Native timing repeat {repeat}/{repeats} finished", flush=True)
    groups = defaultdict(list)
    by_repeat = defaultdict(list)
    for row in rows:
        controller = row["controller"]
        groups[controller].append(row["decision_ns"])
        by_repeat[row["repeat"], controller].append(row["decision_ns"])
        if controller == "MPC":
            groups["MPC QP" if row["qp_solved"] else "MPC analytic"].append(row["decision_ns"])
    write(HERE / "native_timing_summary.csv", [timing.stats(label, values) for label, values in groups.items()])
    write(HERE / "native_timing_by_repeat.csv",
          [dict(repeat=repeat, **timing.stats(label, values)) for (repeat, label), values in by_repeat.items()])
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    metadata = dict(cpu=cpu, logical_cpu=sorted(os.sched_getaffinity(0)), compiler=compiler,
                    variant=variant, max_steering_delta_rad=dict(deviation), parity_samples=dict(samples),
                    numpy=np.__version__, python=sys.version.split()[0], mpc_fallbacks=sum(r["fallback"] for r in rows))
    (HERE / "native_timing_host.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("noise", "moderate"), default="noise")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    load_policies()
    seed = sweep.TRAIN_SEEDS[0]
    sweep.POLICIES["LANS", seed] = PPO.load(HERE / f"{args.variant}_policy_{seed}.zip", device="cpu")
    with TemporaryDirectory() as temporary:
        library = str(Path(temporary) / "native.so")
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(HERE / "native_decision.cpp"),
                        "-o", library], check=True)
        compiler = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
        native = NativeControllers(library)
        try:
            for controller in ("LANS", "PPO"):
                native.register(sweep.POLICIES[controller, seed])
            run_benchmark(native, args.repeats, compiler, args.variant)
        finally:
            native.close()


if __name__ == "__main__":
    main()
