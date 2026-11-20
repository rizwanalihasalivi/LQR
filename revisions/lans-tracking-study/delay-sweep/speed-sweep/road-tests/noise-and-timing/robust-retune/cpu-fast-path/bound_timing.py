"""Benchmark the fixed-delay native controller path with bound scenario inputs."""
import argparse
from collections import defaultdict
from ctypes import Structure, addressof, c_char_p, c_double, c_int, c_void_p, py_object, pythonapi
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory

import numpy as np
from stable_baselines3 import PPO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from native_timing import NativeControllers, timing
from noise_replay import DELAYS, LEVELS, ROADS, SPEEDS, load_policies, sweep, write


GAINS_NAME = b"lans.gains"
ACTOR_NAME = b"lans.actor"


class ActorContext(Structure):
    _fields_ = [("actor", c_void_p), ("gains", c_void_p), ("reference", c_void_p),
                ("transform", c_void_p), ("stride", c_int), ("hybrid", c_int),
                ("speed", c_double), ("residual_scale", c_double), ("gain", c_double)]


class BoundControllers(NativeControllers):
    def __init__(self, library):
        super().__init__(library)
        spec = spec_from_file_location("lans_bound", library)
        self.bound = module_from_spec(spec)
        spec.loader.exec_module(self.bound)
    def bind(self, env, model, optimizer, delay):
        capsule = pythonapi.PyCapsule_New
        capsule.argtypes = [c_void_p, c_char_p, c_void_p]
        capsule.restype = py_object
        if optimizer is not None:
            return lambda: timing.decide(env, None, optimizer, delay)
        # The replay keeps the reference and gain buffers alive across the episode.
        gain_pointer = c_void_p(env.K.ctypes.data)
        if model is None:
            steer = self.bound.lqr
            gains = capsule(gain_pointer, GAINS_NAME, None)

            def command():
                return steer(gains, env.x)

            return command
        hybrid = env.variant == "lans"
        gain = math.exp(-env.beta_decay * max(0., delay - 5.)) if hybrid else 1.
        context = ActorContext(self.actors[model], gain_pointer, env.ref.ctypes.data, self.transform.ctypes.data,
                               env.ref.shape[1], hybrid, env.vx, env.residual_scale, gain)
        context_capsule = capsule(addressof(context), ACTOR_NAME, None)
        steer = self.bound.actor

        def command(owner=context):
            # Own the C struct until the bound callable is released.
            mean_delay = sum(env.history) / len(env.history) if env.history else 0.
            return steer(context_capsule, env.x, env.k, env.last_delay, mean_delay)

        return command


def run_benchmark(controllers, repeats, compiler):
    cells = tuple(product(SPEEDS, DELAYS, ROADS))
    differences = defaultdict(float)
    counts = defaultdict(int)
    noise_counts = defaultdict(int)

    def validate(env, model, mpc, delay):
        decide = controllers.bind(env, model, mpc, delay)
        if mpc is not None:
            return decide

        rng = np.random.default_rng(88210)
        sigmas = tuple(np.array(values) for values in LEVELS.values())

        def check(result):
            difference = abs(result - timing.decide(env, model, mpc, delay))
            if not math.isfinite(difference) or difference >= 5e-6:
                raise RuntimeError(f"{env.variant} steering difference of {difference:.6g} rad")
            differences[env.variant] = max(difference, differences[env.variant])

        def compared():
            result = decide()
            check(result)
            counts[env.variant] += 1
            if counts[env.variant] % 16 == 0:
                clean = env.x
                noise = rng.normal(size=4)
                try:
                    for sigma in sigmas:
                        env.x = clean + noise * sigma
                        check(decide())
                        noise_counts[env.variant] += 1
                finally:
                    env.x = clean
            return result

        return compared

    for cell in cells:
        for _ in timing.scenario(*cell, bind=validate):
            pass
    print("Steering parity verified", dict(differences), flush=True)
    groups = defaultdict(list)
    by_repeat = []
    for repeat in range(1, repeats + 1):
        results = defaultdict(list)
        for cell in cells:
            for row in timing.scenario(*cell, bind=controllers.bind):
                controller = row["controller"]
                elapsed = row["decision_ns"]
                results[controller].append(elapsed)
                if controller == "MPC":
                    groups["MPC QP" if row["qp_solved"] else "MPC analytic"].append(elapsed)
        for label, times in results.items():
            groups[label].extend(times)
            by_repeat.append(dict(repeat=repeat, **timing.stats(label, times)))
        print(f"Timing repeat {repeat}/{repeats} finished", flush=True)

    summary = [timing.stats(label, values) for label, values in groups.items()]
    summary.sort(key=lambda row: (row["controller"] not in sweep.CONTROLLERS, row["median_us"]))
    write(HERE / "timing_summary.csv", summary)
    write(HERE / "timing_by_repeat.csv", by_repeat)
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    metadata = dict(cpu=cpu, logical_cpu=min(os.sched_getaffinity(0)), compiler=compiler,
                    variant="moderate", max_steering_delta_rad=dict(differences), clean_parity_calls=dict(counts), noisy_parity_calls=dict(noise_counts))
    (HERE / "host.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    load_policies()
    seed = sweep.TRAIN_SEEDS[0]
    sweep.POLICIES["LANS", seed] = PPO.load(HERE.parent / f"moderate_policy_{seed}.zip", device="cpu")
    with TemporaryDirectory() as temporary:
        library = str(Path(temporary) / "lans_bound.so")
        flags = ["-O3", "-std=c++17", "-march=native", "-ffp-contract=off",
                 "-DLANS_PACK_FIRST_LAYER", "-shared", "-fPIC", "-I", sysconfig.get_path("include")]
        subprocess.run(["g++", *flags, str(HERE.parent / "native_decision.cpp"),
                        str(HERE / "python_bridge.cpp"), "-o", library], check=True)
        compiler = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
        controllers = BoundControllers(library)
        try:
            for controller in ("LANS", "PPO"):
                controllers.register(sweep.POLICIES[controller, seed])
            run_benchmark(controllers, args.repeats, compiler)
        finally:
            controllers.close()


if __name__ == "__main__":
    main()
