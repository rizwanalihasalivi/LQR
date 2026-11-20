"""Summarize package joules from RAPL counter rows or timed macOS power samples."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import sys

CONTROLLERS = ("LANS", "PPO", "LQR", "MPC")
PHASES = ("IDLE", *CONTROLLERS)
HEADER = re.compile(r"Sampled system activity\s*\((.*?)\)\s*\(([\d.]+)\s*ms elapsed\)", re.I)
POWER = re.compile(r"^\s*(Package Power|Intel energy model derived package power(?:\s*\([^)]*\))?)"
                   r"\s*:\s*([\d.]+)\s*(mW|W)\b", re.I)


def positive(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def read_workloads(paths):
    rows = []
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                name = row["controller"]
                n = int(row["decisions"])
                elapsed = positive(row["elapsed_s"], "elapsed_s")
                start = int(row["start_unix_ns"]) / 1e9
                finish = int(row["end_unix_ns"]) / 1e9
                if name not in PHASES or (name == "IDLE" and n != 0) or (name != "IDLE" and (n <= 0 or n % 18836)):
                    raise ValueError(f"Invalid controller or shared-tape decision count in {path}")
                if abs((finish - start) - elapsed) > .05:
                    raise ValueError(f"Wall and steady clocks disagree in {path}")
                rows.append(dict(controller=name, decisions=n, elapsed_s=elapsed, start=start, finish=finish,
                                 measured=row.get("package_energy_j", "")))
    return rows


def timestamp(value):
    for form in ("%a %b %d %H:%M:%S %Y %z", "%a %b %d %H:%M:%S %Y"):
        try:
            parsed = datetime.strptime(value, form)
            return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized powermetrics timestamp: {value}")


def load_samples(path):
    samples = []
    block = None
    for line in path.read_text().splitlines():
        header = HEADER.search(line)
        if header:
            if block is not None:
                samples.append(block)
            block = [timestamp(header[1]), positive(header[2], "sample duration") / 1e3, None, None]
        power = POWER.match(line)
        if block is not None and power:
            if block[2] is not None:
                raise ValueError(f"Multiple package-power readings per sample in {path}")
            block[2] = float(power[2]) / (1e3 if power[3].lower() == "mw" else 1)
            block[3] = power[1]
    if block is not None:
        samples.append(block)
    if len(samples) < 2 or any(sample[2] is None or not math.isfinite(sample[2]) for sample in samples):
        raise ValueError(f"No complete package-power sample sequence in {path}")
    if len({s[3] for s in samples}) != 1:
        raise ValueError(f"Package-power label changed within {path}")
    return samples


def integrate(samples, start, end):
    # The printed wall timestamps have whole-second precision. Derive allowable
    # offsets from them and the explicitly reported sample durations.
    relative = 0.
    intervals = []
    lower, upper = -math.inf, math.inf
    for stamp, seconds, watts, _ in samples:
        right = relative + seconds
        lower = max(lower, stamp - right)
        upper = min(upper, stamp + 1 - right)
        intervals.append((relative, right, watts))
        relative = right
    if lower > upper:
        raise ValueError("Power sample timestamps and elapsed durations cannot be aligned")
    if upper > start or lower + relative < end:
        raise ValueError("Power samples must bracket the full workload under timestamp rounding")

    def energy(offset):
        return sum(max(0., min(right + offset, end) - max(left + offset, start)) * watts
                   for left, right, watts in intervals)

    # Overlap is piecewise linear in the offset, with extrema at these breaks.
    breakpoints = [lower, upper]
    for left, right, _ in intervals:
        for boundary in (start, end):
            for time in (left, right):
                if lower <= boundary - time <= upper:
                    breakpoints.append(boundary - time)
    bounds = [energy(point) for point in breakpoints]
    return energy((lower + upper) / 2), min(bounds), max(bounds)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["controller"]].append(row)
    if set(grouped) != set(PHASES):
        raise ValueError("A comparison needs IDLE and all four controllers")
    if len({len(grouped[name]) for name in PHASES}) != 1:
        raise ValueError("Use the same number of workload repeats for every phase")
    if len({row["source"] for row in rows}) != 1:
        raise ValueError("All phases must use the same package power or energy source")
    idle = grouped["IDLE"]
    idle_elapsed = sum(r["elapsed_s"] for r in idle)
    idle_power = sum(r["joules"] for r in idle) / idle_elapsed
    idle_min = sum(r["min_j"] for r in idle) / idle_elapsed
    idle_max = sum(r["max_j"] for r in idle) / idle_elapsed
    summary = []
    for name in CONTROLLERS:
        runs = grouped[name]
        count = sum(r["decisions"] for r in runs)
        elapsed = sum(r["elapsed_s"] for r in runs)
        total = sum(r["joules"] for r in runs)
        summary.append(dict(
            controller=name, repeats=len(runs), decisions=count, elapsed_s=elapsed,
            package_energy_j=total, package_j_per_decision=total / count,
            package_j_per_decision_low=sum(r["min_j"] for r in runs) / count,
            package_j_per_decision_high=sum(r["max_j"] for r in runs) / count,
            idle_w=idle_power, idle_adjusted_j_per_decision=(total - idle_power * elapsed) / count,
            idle_adjusted_j_per_decision_low=(sum(r["min_j"] for r in runs) - idle_max * elapsed) / count,
            idle_adjusted_j_per_decision_high=(sum(r["max_j"] for r in runs) - idle_min * elapsed) / count,
            source=runs[0]["source"],
        ))
    return sorted(summary, key=lambda row: row["package_j_per_decision"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rapl", nargs="+", type=Path, help="CSV rows with measured package_energy_j from the same counter")
    source.add_argument("--mac", type=Path, help="Folder from Run-Energy.command")
    args = parser.parse_args()
    if args.rapl:
        rows = read_workloads(args.rapl)
        for row in rows:
            joules = positive(row["measured"], "measured package_energy_j")
            row.update(joules=joules, min_j=joules, max_j=joules, source="RAPL package counter")
    else:
        rows = read_workloads([args.mac / (name + "-workload.csv") for name in PHASES])
        for row in rows:
            samples = load_samples(args.mac / (row["controller"] + "-power.txt"))
            joules, lo, hi = integrate(samples, row["start"], row["finish"])
            row.update(joules=joules, min_j=lo, max_j=hi, source="macOS sampled " + samples[0][3])
    result = summarize(rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(result[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(result)


if __name__ == "__main__":
    main()
