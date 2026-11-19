# Frozen-controller speed sweep

This test uses the final LANS checkpoints from commit `dd878a1`, the study's equal-pretrained PPO checkpoints, the same robust LQR gain and the same objective-matched MPC objective, horizon and steering bounds. MPC uses the known speed in its plant model. All four controllers act through the original full-command FIFO. The state equations and speed observation use the selected speed. Policy weights, LANS gains, failure thresholds, steering saturation and the 100 Hz sample rate are preserved.

The main comparison holds the reference geometry fixed over the same 250 m longitudinal course, with the reference parameters anchored to the published 90 km/h cases. At speed ratio `r = speed / 90`, the reference position is sampled at time `r × t` and derivative order `n` is multiplied by `r^n`. Episode duration is `10/r` seconds. The supplementary 200 ms comparison holds the time-based profile and 10-second duration fixed. Its 90 km/h rows share the identical main runs.

- Speeds are 15, 30, 60, 90, 120, 150 and 200 km/h.
- Fixed delays in the main comparison are 0, 200 and 1000 ms.
- All cases use the same 1650 kg plant mass, four reference paths, two original reference scales and five paired physical seeds. LANS and PPO each use all three policy seeds.
- Each speed-and-delay group contains 120 LANS, 120 PPO, 40 LQR and 40 MPC episodes.
- The independent check repeats the 200 ms fixed-road comparison at 120, 150 and 200 km/h on physical seeds 33800–33804.
- Rank uses mean lateral RMSE with the existing remaining-course penalty for early termination. Completion is reported separately.
- The model is a linear bicycle simulation with no nonlinear tire grip limit. Reference lateral acceleration is recorded in the episode CSV so that aggressive high-speed cases can be identified.

Run from this directory with the tracking-study dependencies.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python run.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python run.py --fresh-high-speed
```

[Per-speed ranks](rank_by_speed_and_delay.csv) · [Per-path ranks](rank_by_path.csv) · [Episode data](episodes.csv) · [Independent high-speed results](fresh_rank_by_speed_and_delay.csv)

## Verified results

Cells report LANS rank and completions out of 120 on the fixed road.

| Speed | 0 ms | 200 ms | 1000 ms |
| ---: | --- | --- | --- |
| 15 km/h | 2, 120/120 | 4, 75/120 | 2, 0/120 |
| 30 km/h | 2, 120/120 | 2, 105/120 | 2, 0/120 |
| 60 km/h | 2, 120/120 | 1, 120/120 | 2, 0/120 |
| 90 km/h | 1, 120/120 | 1, 120/120 | 1, 80/120 |
| 120 km/h | 2, 120/120 | 2, 0/120 | 2, 0/120 |
| 150 km/h | 2, 120/120 | 3, 0/120 | 1, 0/120 |
| 200 km/h | 3, 90/120 | 3, 0/120 | 1, 0/120 |

All 360 LANS episodes at 90 km/h matched the published checkpoint replay exactly at 0, 200 and 1000 ms. The main and supplementary comparisons ran 8,640 distinct episodes. The independent confirmation ran another 960.

At 200 ms, LQR ranked first at 120, 150 and 200 km/h on both seed splits. In the independent split LANS ranked third and completed zero episodes at all three speeds. The supplementary fixed-time 200 ms cases also yielded zero LANS completions at and above 120 km/h.

At 1000 ms and 150 or 200 km/h, rank 1 means the lowest failure-penalized RMSE among four controllers with zero completions. At 200 km/h and zero delay, all 90 LANS runs on DLC, smooth step and sine completed. All 30 LANS slalom runs terminated. The 200 km/h fixed-road slalom reached approximately 101 m/s² peak reference lateral acceleration in the more demanding scale.
