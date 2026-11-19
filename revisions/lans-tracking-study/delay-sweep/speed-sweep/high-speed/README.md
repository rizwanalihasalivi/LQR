# High-speed LANS refinement

This study extends the low-speed checkpoint set from commit `fb006c9` to 120, 150 and 200 km/h at 0, 200 and 1000 ms. The original 250 m course, four reference paths, full-command FIFO, tire model, steering range, termination thresholds and baseline episodes are reused. The controller retains the original 11 observations and 64×64 actor and adjusts weights using the already available speed, previous delay and preview channels. The tested points are discrete speed-delay combinations.

The delayed coefficients are fit on disjoint simulation seeds. Nominal refinement uses bounded offline steering targets, frequency-response initialization and closed-loop steering sensitivities. The additional 200 km/h curvature response uses measured preview features to identify high curvature and stays within the existing actor. The training label generator can see future reference values. Deployed deterministic inference uses the original observations.

The retained `gn` checkpoint set preserves the original first-layer encoding for the 200 and 1000 ms delay gates. Experimental `extra` and `split` fits reuse features assigned to other delays and are saved for research. They missed the nominal benchmark target. The archived delay-sweep checkpoints remain available.

## Original benchmark

The retained set ranks first in 7/9 high-speed groups by mean failure-penalized lateral RMSE. The other two groups retain all 120 completions.

| Speed | 0 ms | 200 ms | 1000 ms |
| ---: | --- | --- | --- |
| 120 km/h | #2, 120/120 | #1, 120/120 | #1, 34/120 |
| 150 km/h | #2, 120/120 | #1, 120/120 | #1, 18/120 |
| 200 km/h | #1, 120/120 | #1, 90/120 | #1, 1/120 |

LANS completes 743/1080 original episodes, up from 330/1080 for the published speed sweep. At zero delay, MPC's lead over the retained candidate is approximately 7.51 micrometers at 120 km/h and 33.0 micrometers at 150 km/h. The early error remains above the offline steering lower bound. The independent rerun also returns 7/9 first places.

The LP feasibility check allows arbitrary future-informed bounded steering after the forced FIFO prefix. Its completion upper bounds are 90/120 at 120 and 150 km/h with 1000 ms, 90/120 at 200 km/h with 200 ms and 75/120 at 200 km/h with 1000 ms. Passing the relaxation establishes that bounded steering is physically feasible in the linear simulator for that initial state. Achieving the bound with the causal actor remains a separate test.

- [Original ranking](gn_benchmark_ranking.csv) and [episodes](gn_benchmark_episodes.csv)
- [Independent ranking](gn_fresh_ranking.csv) and [episodes](gn_fresh_episodes.csv)
- [Early error and lower bound](nominal_error_phases.csv)
- [Physical feasibility](authority.csv)

## Reproduce

Run from this directory with the tracking-study dependencies. The selected coefficient files are included.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
g++ -O3 -std=c++17 -shared -fPIC ../low-speed/rollout.cpp -o rollout.so
python authority.py
python fit_high.py nominal
python fit_high.py delayed
python nominal_teacher.py
python slalom.py
# The stored nominal_combined.json uses the teacher at 120 and 150 and the initial ordinary response at 200.
python harmonic.py
python nominal_gn.py
python phase.py
python replay_high.py --variant gn
python replay_high.py --variant gn --split fresh
python replay_high.py --variant gn --split guard --guard-speeds 90
python replay_high.py --variant gn --split guard --guard-speeds 15 30 60
```

## Preservation checks

The retained high-speed checkpoints keep rank 1 in all 12 original benchmark groups at 15, 30, 60 and 90 km/h. At 90 km/h and 1000 ms they complete 83/120 episodes in the guard replay. The lower-speed archive is unchanged.

The separate startup/steady-state fit narrowed the 120 km/h no-delay gap to 3.73 micrometers in its benchmark replay. Its 150 km/h nominal RMSE was higher than the retained checkpoint's. Both remained rank 2.

- [15, 30 and 60 km/h guard](gn_guard_15-30-60_ranking.csv)
- [90 km/h guard](gn_guard_90_ranking.csv)
- [Retained checkpoint hashes](retained_checkpoints.sha256)
