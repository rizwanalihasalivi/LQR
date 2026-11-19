"""Measure mixed-delay ticks with queued work absent from the actor's delay channels."""
import csv
from collections import defaultdict

import numpy as np

from distill import HERE
from batch_sim import build_cases, prepare
from calibration import ADDITIONAL_SPLITS


def main():
    out = HERE / "queue_visibility.csv"
    results = defaultdict(lambda: np.zeros(6, dtype=int))
    ticks = np.arange(1000)
    for split, (seeds, scales) in ADDITIONAL_SPLITS.items():
        cases = [c for c in build_cases(seeds, scales) if c["delay"] == "mixed"]
        for case, trace in zip(cases, prepare(cases)["delays"], strict=True):
            delay = trace[:1000]
            release = np.maximum.accumulate(ticks + np.ceil(delay / 10).astype(int))
            prev_release = np.r_[-1, release[:-1]]
            observed = np.zeros_like(delay, dtype=bool)
            for tick in range(1, len(delay)):
                observed[tick] = np.any(delay[max(0, tick-10):tick] > 0)
            queued = prev_release > ticks
            blind = queued & ~observed
            unattenuated = blind & (delay <= 5)
            results[split, case["path"]] += [
                1, len(delay), queued.sum(), blind.sum(), unattenuated.sum(),
                int(unattenuated.any()),
            ]
    with out.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "path", "episodes", "ticks", "pending_queue_ticks",
                         "zero_observed_delay_pending_ticks", "unattenuated_pending_ticks",
                         "episodes_with_unattenuated_pending"))
        for key, counts in sorted(results.items()):
            writer.writerow((*key, *counts))


if __name__ == "__main__":
    main()
