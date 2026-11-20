"""Replay dedicated road references and the archived lane-change benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from itertools import product
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO
import torch

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parent
RECOVERY = SWEEP / "high-speed" / "nominal-recovery"
sys.path[:0] = [str(SWEEP), str(SWEEP / "low-speed")]

from batched import run
import run as sweep

NEW_ROADS = ("straight", "curve")
LANE_CHANGES = ("step", "dlc")
FRESH_SEEDS = tuple(range(48200, 48205))
original_configure = sweep.configure_env


def configure(controller, path, delay, speed, amplitude, scale, profile):
    env, horizon, ref = original_configure(controller, path, delay, speed, amplitude, scale, profile)
    if path == "straight":
        ref.fill(0.)
    elif path == "curve":
        # Signed constant spatial second derivative after a smooth entry ramp.
        # At nominal scale this bends the reference tangent by about 11 degrees.
        vx = speed / 3.6
        s = vx * .01 * np.arange(horizon + 6) * scale
        entry = 30.
        z = np.minimum(s / entry, 1.)
        tail = np.maximum(s - entry, 0.)
        curvature = amplitude / 1200.
        rate = vx * scale
        ref = curvature * np.array([
            entry**2 * (z**4 / 4 - z**5 / 10) + entry * tail / 2 + tail**2 / 2,
            rate * (entry * (z**3 - z**4 / 2) + tail),
            rate**2 * (3*z**2 - 2*z**3),
            rate**3 / entry * (6*z - 6*z**2),
        ])
    return env, horizon, ref


def archived_lane_changes(speeds, delays):
    with (RECOVERY / "switching_guard_episodes.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["path"] not in LANE_CHANGES:
                continue
            for key in ("speed_kmh", "delay_ms", "completed", "mpc_fallbacks"):
                row[key] = int(row[key])
            if row["speed_kmh"] not in speeds or row["delay_ms"] not in delays:
                continue
            for key in ("rmse_penalized_m", "rmse_observed_m", "saturation_fraction",
                        "applied_saturation_fraction"):
                row[key] = float(row[key])
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--speeds", nargs="+", type=int, choices=sweep.SPEEDS, default=sweep.SPEEDS)
    parser.add_argument("--delays", nargs="+", type=int, choices=sweep.ROAD_DELAYS, default=sweep.ROAD_DELAYS)
    parser.add_argument("--tuned", nargs="?", const="road", choices=("road", "extended"))
    parser.add_argument("--roads", nargs="+", choices=NEW_ROADS + LANE_CHANGES)
    parser.add_argument("--lans-only", action="store_true")
    parser.add_argument("--fit-cells", action="store_true")
    args = parser.parse_args()
    if (args.lans_only or args.fit_cells) and not args.tuned:
        parser.error("--lans-only and --fit-cells require --tuned")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sweep.configure_env = configure
    sweep.HERE = HERE
    if args.fresh:
        sweep.sweep.P["test"]["seeds"] = FRESH_SEEDS
    for seed in sweep.TRAIN_SEEDS:
        policy_path = HERE / f"{args.tuned}_policy_{seed}.zip" if args.tuned else RECOVERY / f"switching_policy_{seed}.zip"
        sweep.POLICIES["LANS", seed] = PPO.load(policy_path, device="cpu")
        if not args.lans_only:
            sweep.POLICIES["PPO", seed] = sweep.sweep.load("limited_ppo", seed)
    roads = args.roads or (NEW_ROADS + LANE_CHANGES if args.fresh or args.tuned else NEW_ROADS)
    controllers = ("LANS",) if args.lans_only else sweep.CONTROLLERS
    cells = list(product(args.speeds, args.delays))
    if args.fit_cells:
        records = json.loads((HERE / f"{args.tuned}_fit.json").read_text())
        fitted = {(record["speed"], record["delay"]) for record in records}
        cells = [cell for cell in cells if cell in fitted]
        if not cells:
            parser.error("No fitted cells match the requested speeds and delays")
    jobs = [
        ("fixed_road", speed, delay, controller, seed, path)
        for (speed, delay), path, controller in product(cells, roads, controllers)
        for seed in (sweep.TRAIN_SEEDS if controller in ("LANS", "PPO") else (-1,))
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for index, batch in enumerate(pool.map(run, jobs, chunksize=1), 1):
            rows.extend(batch)
            if index % 48 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} jobs finished", flush=True)
    expected = len(jobs) * len(sweep.sweep.P["test"]["seeds"]) * len(sweep.sweep.P["test"]["scales"])
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} episodes, found {len(rows)}")
    prefix = "fresh" if args.fresh else "original"
    reference_prefix = prefix
    if args.tuned:
        prefix = ("tuned_" if args.tuned == "road" else "extended_") + prefix
    for option, values in (("speeds", args.speeds), ("delays", args.delays), ("roads", args.roads)):
        if values and f"--{option}" in sys.argv:
            prefix += "_" + "-".join(map(str, values))
    if args.fit_cells:
        prefix += "_fit_cells"
    if args.lans_only:
        cases = {(r["speed_kmh"], r["delay_ms"], r["path"]) for r in rows}
        with (HERE / f"{reference_prefix}_episodes.csv").open() as stream:
            baseline = list(csv.DictReader(stream))
        if not args.fresh:
            baseline.extend(archived_lane_changes(args.speeds, args.delays))
        for row in baseline:
            key = (int(row["speed_kmh"]), int(row["delay_ms"]), row["path"])
            if key in cases and row["controller"] != "LANS":
                for name in ("speed_kmh", "delay_ms", "completed", "mpc_fallbacks"):
                    row[name] = int(row[name])
                for name in ("rmse_penalized_m", "rmse_observed_m", "saturation_fraction",
                             "applied_saturation_fraction"):
                    row[name] = float(row[name])
                rows.append(row)
        counts = {}
        for row in rows:
            key = (row["speed_kmh"], row["delay_ms"], row["path"], row["controller"])
            counts[key] = counts.get(key, 0) + 1
        per_policy = len(sweep.sweep.P["test"]["seeds"]) * len(sweep.sweep.P["test"]["scales"])
        for speed, delay, path in cases:
            for controller in sweep.CONTROLLERS:
                expected_count = per_policy * (len(sweep.TRAIN_SEEDS) if controller in ("LANS", "PPO") else 1)
                if counts.get((speed, delay, path, controller)) != expected_count:
                    raise ValueError(f"Invalid episode count for {(speed, delay, path, controller)}")
    sweep.save(f"{prefix}_episodes.csv", rows)
    if not args.fresh and not args.tuned and args.roads is None:
        rows.extend(archived_lane_changes(args.speeds, args.delays))
    sweep.save(f"{prefix}_ranking.csv", sweep.summarize(rows, by_path=True))


if __name__ == "__main__":
    main()
