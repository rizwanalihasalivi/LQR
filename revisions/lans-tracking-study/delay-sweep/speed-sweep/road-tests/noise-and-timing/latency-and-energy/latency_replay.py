"""Replay measured decision durations through a modeled sensor-to-command pipeline."""
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
PERIOD_NS = 10_000_000  # Existing 100 Hz simulator.
CONTROLLERS = {"LANS", "PPO", "LQR", "MPC"}


def quantile(sorted_values, q):
    index = q * (len(sorted_values) - 1)
    lo = int(index)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (index - lo) * (sorted_values[hi] - sorted_values[lo])


def replay(frames, delay_ms):
    """Return (generation, ready latency, next actuator tick latency) samples."""
    transport_ns = delay_ms * 1_000_000
    compute_end = last_release = 0
    ready = []
    latest_by_actuator_tick = {}
    for sample in sorted(frames):
        sensor_ns = sample * PERIOD_NS
        compute_end = max(sensor_ns, compute_end) + frames[sample]
        release = max(last_release, compute_end + transport_ns)
        last_release = release
        actuator_tick = (release + PERIOD_NS - 1) // PERIOD_NS
        ready.append(release - sensor_ns)
        # The FIFO consumes all ready packets at a tick and applies the newest.
        latest_by_actuator_tick[actuator_tick] = (sample, actuator_tick * PERIOD_NS - sensor_ns)
    return ready, list(latest_by_actuator_tick.values())


def summarize(ready, applied, base, speed):
    result = {**base, "samples": len(ready), "applied": len(applied), "superseded": len(ready) - len(applied)}
    for name, values in (("ready", ready), ("apply", applied)):
        values = sorted(values)
        for label, ns in (("median", quantile(values, .50)), ("p95", quantile(values, .95)),
                          ("p99", quantile(values, .99)), ("mean", mean(values)), ("max", values[-1])):
            result[f"{name}_{label}_ms"] = ns / 1e6
            result[f"{name}_{label}_m"] = ns / 1e9 * (speed / 3.6)
    return result


def save(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    trials = defaultdict(dict)
    with (HERE.parent / "timing_decisions.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(int(row[k]) for k in ("repeat", "speed_kmh", "delay_ms")) + (row["path"], row["controller"])
            tick, ns = int(row["tick"]), int(row["decision_ns"])
            if row["controller"] not in CONTROLLERS or ns < 0 or tick in trials[key]:
                raise ValueError(f"Invalid timing row for {key} at tick {tick}")
            trials[key][tick] = ns
    cells = defaultdict(lambda: ([], []))
    pooled = defaultdict(lambda: ([], []))
    for key, frames in sorted(trials.items()):
        repeat, speed, delay, path, controller = key
        if set(frames) != set(range(len(frames))):
            raise ValueError(f"Non-contiguous timing trace for {key}")
        available, events = replay(frames, delay)
        applied = [ns for _, ns in events]
        for group, target in ((cells, (speed, delay, path, controller)),
                              (pooled, (speed, delay, controller))):
            ready_list, apply_list = group[target]
            ready_list.extend(available)
            apply_list.extend(applied)
    save("distance_by_cell.csv", [
        summarize(ready, applied, dict(speed_kmh=speed, delay_ms=delay, path=path, controller=controller), speed)
        for (speed, delay, path, controller), (ready, applied) in sorted(cells.items())
    ])
    save("distance_by_speed_delay.csv", [
        summarize(ready, applied, dict(speed_kmh=speed, delay_ms=delay, controller=controller), speed)
        for (speed, delay, controller), (ready, applied) in sorted(pooled.items())
    ])
    print(f"Replayed {sum(len(frames) for frames in trials.values()):,} measured decisions in {len(cells)} cells.")


if __name__ == "__main__":
    main()
