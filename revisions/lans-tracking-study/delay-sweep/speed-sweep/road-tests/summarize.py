"""Rank each road by equally weighting the speed-delay cells in a full replay."""
import argparse
import csv
from pathlib import Path
from statistics import mean

PATHS = ("straight", "step", "dlc", "curve")


def summarize(rows):
    cells = {(r["speed_kmh"], r["delay_ms"]) for r in rows}
    result = []
    for path in PATHS:
        scores = []
        for controller in ("LANS", "PPO", "LQR", "MPC"):
            selected = [r for r in rows if r["path"] == path and r["controller"] == controller]
            if len(selected) != len(cells) or {(r["speed_kmh"], r["delay_ms"]) for r in selected} != cells:
                raise ValueError(f"Incomplete or duplicate cells for {path}/{controller}")
            scores.append(dict(
                path=path,
                controller=controller,
                mean_penalized_rmse_m=mean(float(r["rmse_penalized_m"]) for r in selected),
                cells=len(cells),
                first_places=sum(int(r["rank"]) == 1 for r in selected),
                completed=sum(int(r["completed"]) for r in selected),
                episodes=sum(int(r["episodes"]) for r in selected),
            ))
        scores.sort(key=lambda r: r["mean_penalized_rmse_m"])
        for row in scores:
            row["rank"] = 1 + sum(other["mean_penalized_rmse_m"] < row["mean_penalized_rmse_m"] - 1e-12
                                  for other in scores)
        result.extend(scores)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rankings", nargs="+", type=Path)
    for path in parser.parse_args().rankings:
        with path.open(newline="") as stream:
            rows = summarize(list(csv.DictReader(stream)))
        filename = path.stem.removesuffix("_ranking") + "_by_road.csv"
        with path.with_name(filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
