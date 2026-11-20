"""Score real sensor-to-steering measurements joined by hardware command ID."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
from statistics import mean
import sys

from latency_replay import CONTROLLERS, quantile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="Joined synchronized sensor and physical actuator event CSV")
    args = parser.parse_args()
    measurements = defaultdict(list)
    keys = set()
    with args.events.open(newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["controller"]
            key = (name, row["run_id"], row["command_id"])
            sensor, actuator = (int(row[k]) for k in ("sensor_ns", "actuator_ns"))
            before, after = (float(row[k]) for k in ("sensor_odometer_m", "actuator_odometer_m"))
            if (name not in CONTROLLERS or key in keys or sensor < 0 or actuator < sensor or
                    not math.isfinite(before) or not math.isfinite(after) or after < before):
                raise ValueError(f"Invalid or duplicate physical event: {key}")
            keys.add(key)
            measurements[name].append(((actuator - sensor) / 1e6, after - before))
    if set(measurements) != CONTROLLERS:
        raise ValueError("The physical comparison needs all four controllers")
    rows = []
    for name, pairs in sorted(measurements.items()):
        row = dict(controller=name, applied_commands=len(pairs))
        for columns, unit in zip(zip(*pairs), ("ms", "m"), strict=True):
            values = sorted(columns)
            for statistic, value in (("median", quantile(values, .5)), ("p95", quantile(values, .95)),
                                     ("p99", quantile(values, .99)), ("mean", mean(values)), ("max", values[-1])):
                row[f"{statistic}_{unit}"] = value
        rows.append(row)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
