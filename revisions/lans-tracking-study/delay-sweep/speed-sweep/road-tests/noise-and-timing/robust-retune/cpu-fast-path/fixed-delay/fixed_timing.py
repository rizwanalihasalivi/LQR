"""Benchmark fixed-delay activation plans on the existing common-state replay."""
import argparse
from collections import defaultdict
from ctypes import POINTER, byref, c_char_p, c_double, c_int, c_uint, c_void_p, py_object, pythonapi
from itertools import product
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
from bound_timing import ActorContext, BoundControllers, run_benchmark, timing
from noise_replay import DELAYS, ROADS, SPEEDS, load_policies, sweep, write

CAPSULE_NAME = b"lans.fixed"


class FixedControllers(BoundControllers):
    def __init__(self, library):
        super().__init__(library)
        self.library.create_fixed_actor.argtypes = [POINTER(ActorContext), c_double]
        self.library.create_fixed_actor.restype = c_void_p
        self.library.destroy_fixed_actor.argtypes = [c_void_p]
        self.library.destroy_fixed_actor.restype = None
        self.library.fixed_actor_rows.argtypes = [c_void_p, c_int]
        self.library.fixed_actor_rows.restype = c_uint
        self.plans = []
        self.rows = {}

    def plan(self, env, model, delay):
        hybrid = env.variant == "lans"
        gain = math.exp(-env.beta_decay * max(0., delay - 5.)) if hybrid else 1.
        context = ActorContext(self.actors[model], env.K.ctypes.data, env.ref.ctypes.data,
                               self.transform.ctypes.data, env.ref.shape[1], hybrid,
                               env.vx, env.residual_scale, gain)
        plan = self.library.create_fixed_actor(byref(context), delay)
        self.plans.append(plan)
        self.rows[env.variant, round(env.vx * 3.6), delay] = [
            self.library.fixed_actor_rows(plan, phase) for phase in (0, 1)
        ]
        return plan

    def bind(self, env, model, optimizer, delay):
        if model is None:
            return super().bind(env, model, optimizer, delay)
        capsule = pythonapi.PyCapsule_New
        capsule.argtypes = [c_void_p, c_char_p, c_void_p]
        capsule.restype = py_object
        context = capsule(self.plan(env, model, delay), CAPSULE_NAME, None)
        steer = self.bound.fixed_actor
        return lambda: steer(context, env.x, env.k)

    def close(self):
        for plan in self.plans:
            self.library.destroy_fixed_actor(plan)
        super().close()


def compile_library(directory):
    library = str(Path(directory) / "lans_bound.so")
    options = ["-O3", "-std=c++17", "-march=native", "-ffp-contract=off",
               "-DLANS_PACK_FIRST_LAYER", "-DLANS_FIXED_DELAY_PLAN", "-shared", "-fPIC",
               "-I", sysconfig.get_path("include")]
    subprocess.run(["g++", *options, str(HERE.parents[1] / "native_decision.cpp"),
                    str(HERE.parent / "python_bridge.cpp"), "-o", library], check=True)
    return library


def benchmark_core(controllers, repeats):
    float_buffer = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    ns_buffer = np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS")
    measure = controllers.library.time_native_core
    measure.argtypes = [c_void_p] * 3 + [float_buffer, c_int, ns_buffer, float_buffer]
    measure.restype = None
    groups = defaultdict(list)
    max_delta = 0.
    for speed, delay, path in product(SPEEDS, DELAYS, ROADS):
        environments, frames = {}, {}

        def capture(env, model, optimizer, fixed_delay):
            environments[env.variant] = env
            if env.variant != "lans":
                return lambda: None

            def record():
                frames[env.k] = env.x.copy()

            return record

        for _ in timing.scenario(speed, delay, path, bind=capture):
            pass
        states = np.ascontiguousarray([frames[i] for i in range(len(frames))])
        lans = controllers.plan(environments["lans"], sweep.POLICIES["LANS", sweep.TRAIN_SEEDS[0]], delay)
        ppo = controllers.plan(environments["ppo"], sweep.POLICIES["PPO", sweep.TRAIN_SEEDS[0]], delay)
        gains = environments["lqr"].K.ctypes.data
        elapsed = np.empty((len(states), 4), dtype=np.uint64)
        commands = np.empty((len(states), 4))
        args = (lans, ppo, gains, states, len(states), elapsed, commands)
        measure(*args)  # Warm each native path on this tape.
        for _ in range(repeats):
            measure(*args)
            max_delta = max(max_delta, float(np.max(np.abs(commands[:, 0] - commands[:, 3]))))
            for index, name in enumerate(("LANS packed core", "PPO core", "LQR core", "LANS previous core")):
                groups[name].extend(elapsed[:, index])
    if not math.isfinite(max_delta) or max_delta >= 5e-6:
        raise RuntimeError(f"Native core command mismatch: {max_delta:.6g} rad")
    ordered = sorted((timing.stats(name, values) for name, values in groups.items()),
                     key=lambda row: row["median_us"])
    write(HERE / "native_core_timing.csv", ordered)
    print("Native core timing finished", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    load_policies()
    seed = sweep.TRAIN_SEEDS[0]
    sweep.POLICIES["LANS", seed] = PPO.load(HERE.parents[1] / f"moderate_policy_{seed}.zip", device="cpu")
    with TemporaryDirectory() as directory:
        controllers = FixedControllers(compile_library(directory))
        try:
            for controller in ("LANS", "PPO"):
                controllers.register(sweep.POLICIES[controller, seed])
            compiler = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
            run_benchmark(controllers, args.repeats, compiler, output=HERE)
            benchmark_core(controllers, args.repeats)
            write(HERE / "activation_rows.csv", [
                dict(controller=kind.upper(), speed_kmh=speed, delay_ms=delay,
                     reset_second_layer_rows=sizes[0], steady_second_layer_rows=sizes[1])
                for (kind, speed, delay), sizes in sorted(controllers.rows.items())
            ])
        finally:
            controllers.close()


if __name__ == "__main__":
    main()
