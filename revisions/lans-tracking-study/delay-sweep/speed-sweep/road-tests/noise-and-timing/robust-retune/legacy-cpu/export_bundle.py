"""Export the frozen timing tape and portable compiler inputs for older Intel hosts."""
import argparse
from importlib.metadata import distribution
from itertools import product
from pathlib import Path
import shutil
import sys

import numpy as np
from stable_baselines3 import PPO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from native_timing import timing
from noise_replay import DELAYS, ROADS, SPEEDS, load_policies, sweep
from features import T

CONTROLLERS = ("LANS", "PPO", "LQR", "MPC")


def binary(stream, data, dtype="<f8"):
    stream.write(np.asarray(data, dtype=dtype).tobytes(order="C"))


def capture_case(speed, delay, path):
    frames = {}
    environments = {}

    def bind(env, model, mpc, delay_ms):
        label = env.variant.upper()
        environments[label] = env

        def decide():
            command = timing.decide(env, model, mpc, delay_ms)
            row = frames.setdefault(env.k, {})
            if label == "LANS":
                row["state"] = env.x.copy()
                row["previous"] = env.prev_u
            row[label] = command
            return command

        return decide

    for _ in timing.scenario(speed, delay, path, bind=bind):
        pass
    tape = np.asarray([
        np.r_[frames[k]["state"], frames[k]["previous"], [frames[k][c] for c in CONTROLLERS]]
        for k in range(len(frames))
    ])
    if tape.ndim != 2 or tape.shape[1] != 9 or not np.isfinite(tape).all():
        raise RuntimeError("The common-state replay produced an invalid timing tape")
    return environments, tape


def copy_license(bundle):
    for file in distribution("osqp").files or ():
        if file.name.lower().startswith("license") and ".dist-info" in str(file):
            shutil.copyfile(file.locate(), bundle / "OSQP-LICENSE")
            return
    raise RuntimeError("OSQP source distribution license was unavailable")


def export(bundle):
    bundle.mkdir(parents=True)
    (bundle / "native/cpu-fast-path/fixed-delay").mkdir(parents=True)
    shutil.copy2(ROOT / "native_decision.cpp", bundle / "native")
    shutil.copy2(ROOT / "cpu-fast-path/fixed-delay/activation_plan.hpp",
                 bundle / "native/cpu-fast-path/fixed-delay")
    for name in ("benchmark.cpp", "build.sh", "run.sh", "README.md"):
        shutil.copy2(HERE / name, bundle / name)
    copy_license(bundle)
    load_policies()
    seed = sweep.TRAIN_SEEDS[0]
    sweep.POLICIES["LANS", seed] = PPO.load(ROOT / f"moderate_policy_{seed}.zip", device="cpu")
    cells = tuple(product(SPEEDS, DELAYS, ROADS))
    with (bundle / "tape.bin").open("wb") as stream:
        stream.write(b"LQRCPU01")
        binary(stream, [len(SPEEDS), len(cells)], "<u4")
        for name in ("LANS", "PPO"):
            policy = sweep.POLICIES[name, seed].policy
            first, _, second, _ = policy.mlp_extractor.policy_net
            for layer in (first, second, policy.action_net):
                for parameter in (layer.weight, layer.bias):
                    binary(stream, parameter.detach().cpu().numpy(), "<f4")
        binary(stream, T)
        for index, speed in enumerate(SPEEDS):
            solver = sweep.SweepMPC(mass=1650., vx=speed / 3.6)
            solver.solver.codegen(str(bundle / "qp"), prefix=f"mpc{speed}_",
                                  extension_name=None, include_codegen_src=index == 0)
            binary(stream, [speed], "<u4")
            for field in ("A", "G", "H", "T", "SW", "inv"):
                binary(stream, getattr(solver, field))
        count = 0
        for speed, delay, path in cells:
            env, frames = capture_case(speed, delay, path)
            lans, ppo, lqr = (env[name] for name in ("LANS", "PPO", "LQR"))
            binary(stream, [speed, delay, len(frames), lans.ref.shape[1]], "<u4")
            binary(stream, [lans.residual_scale, ppo.residual_scale, lans.beta_decay])
            binary(stream, lans.K)
            binary(stream, lqr.K)
            binary(stream, lans.ref)
            binary(stream, frames)
            count += len(frames)
    print(f"Exported {len(cells)} cases and {count} shared states to {bundle}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for the self-contained source bundle")
    export(parser.parse_args().output.resolve())
