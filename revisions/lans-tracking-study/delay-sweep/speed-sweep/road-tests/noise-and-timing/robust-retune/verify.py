"""Compare retuned checkpoints to frozen baselines in paired episode replays."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from itertools import product
from pathlib import Path
import sys

from stable_baselines3 import PPO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from noise_replay import DELAYS, FRESH_SEEDS, LEVELS, ROADS, SPEEDS, load_policies, replay, summarize, sweep, write

LEVELS["clean"] = (0.,) * 4
INDEPENDENT = tuple(range(70200, 70205))


def archived():
    with (HERE.parent / "noise_episodes.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["controller"] == "LANS":
                continue
            for key in ("speed_kmh", "delay_ms", "mpc_fallbacks", "completed"):
                row[key] = int(row[key])
            for key in ("rmse_penalized_m", "rmse_observed_m", "saturation_fraction",
                        "applied_saturation_fraction"):
                row[key] = float(row[key])
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent", action="store_true")
    parser.add_argument("--variant", default="noise", choices=("noise", "moderate"))
    args = parser.parse_args()
    # The shared noise replay lists clean separately when it summarizes results.
    load_policies()
    for seed in sweep.TRAIN_SEEDS:
        sweep.POLICIES["LANS", seed] = PPO.load(HERE / f"{args.variant}_policy_{seed}.zip", device="cpu")
    sweep.sweep.P["test"]["seeds"] = INDEPENDENT if args.independent else FRESH_SEEDS
    controllers = sweep.CONTROLLERS if args.independent else ("LANS",)
    rows = [] if args.independent else list(archived())
    jobs = [(noise, speed, delay, road, controller, seed)
            for noise, speed, delay, road, controller in product(LEVELS, SPEEDS, DELAYS, ROADS, controllers)
            for seed in (sweep.TRAIN_SEEDS if controller in ("LANS", "PPO") else (-1,))]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(replay, job) for job in jobs]
        for i, task in enumerate(as_completed(futures), 1):
            rows.extend(task.result())
            if i % 48 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} replay jobs finished", flush=True)
    rows.sort(key=lambda row: (row["noise"], row["speed_kmh"], row["delay_ms"], row["path"],
                               row["controller"], int(row["policy_seed"]), float(row["amplitude"]),
                               int(row["physical_seed"])))
    keys = {(r["noise"], r["speed_kmh"], r["delay_ms"], r["path"], r["controller"],
             int(r["policy_seed"]), float(r["amplitude"]), int(r["physical_seed"])) for r in rows}
    assert len(keys) == len(rows) == 5760
    LEVELS.pop("clean")
    scores, ranking = summarize(rows)
    prefix = f"{args.variant}_" + ("independent" if args.independent else "paired")
    write(HERE / f"{prefix}_episodes.csv", rows)
    write(HERE / f"{prefix}_by_cell.csv", scores)
    write(HERE / f"{prefix}_summary.csv", ranking)


if __name__ == "__main__":
    main()
