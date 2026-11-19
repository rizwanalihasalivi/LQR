# Low-speed LANS refinement

The target group contains 15, 30 and 60 km/h at constant FIFO delays of 0, 200 and 1000 ms. The evaluator uses the same 250 m course, physical plant mass, reference scales, seeds, steering limit and failure thresholds as the published speed sweep. Reference results for LQR, MPC and PPO on the original benchmark are reused unchanged.

The three new checkpoints start from the retained fixed-delay LANS models. The actor still has the existing 11 observations and 64×64 ReLU architecture. New weights use the existing speed and observed-delay channels to tune the responses at the requested test points. Some existing first-layer delay features are re-expressed algebraically to free neurons. The archived checkpoints and prior episodes are preserved. The new checkpoint set is designed for these measured speed-delay combinations.

The fit uses disjoint physical training and calibration seeds. The first stage fits nominal steering. The delayed fit uses the same FIFO and the nominal actor's reset steering while fitting 200 and 1000 ms responses. Follow-up nominal fits use more initial states and optimize early response and steady-state feedforward separately. The final fit uses spare second-layer units for the low-error correction at 15 km/h without delay. The bounded steering teacher generates offline targets only. All scored LANS episodes use deterministic PPO inference and the original scalar simulator. The final sweep batches policy inference across initial states. The closest 15 km/h condition is additionally checked with single-episode inference. The independent early-error bound allows arbitrary bounded steering and omits error beyond the first 80 updates, so it is optimistic for every real policy.

Run from this directory with the tracking-study Python dependencies.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
g++ -std=c++17 -O3 -shared -fPIC rollout.cpp -o rollout.so
python fit.py nominal
python fit.py delayed
python fit.py polish
python nominal_bound.py
python precision.py
python tail.py
python replay.py
python replay.py --variant polished
python replay.py --variant polished --split anchor
python replay.py --variant precision --speeds 15
python replay.py --variant tail --speeds 15 --delays 0
python replay.py --variant tail --batched
python replay.py --variant tail --split fresh --batched
```

## Original benchmark

The final `tail` checkpoint set ranks first by mean failure-penalized RMSE in all nine requested speed-delay groups. Each group averages DLC, smooth step, sine and slalom. The frozen comparators' aggregate RMSE and completion counts match the original speed-sweep data.

| Speed | 0 ms | 200 ms | 1000 ms |
| ---: | --- | --- | --- |
| 15 km/h | #1, 120/120 | #1, 120/120 | #1, 120/120 |
| 30 km/h | #1, 120/120 | #1, 120/120 | #1, 120/120 |
| 60 km/h | #1, 120/120 | #1, 120/120 | #1, 90/120 |

The combined completion count is 1,050/1,080. All 30 terminations at 60 km/h and 1000 ms are from slalom. Six occurred before the first delayed command could affect the plant. Nominal first-place margins against MPC range from 0.41 to 0.82 micrometers. At 15 km/h the single-episode inference check reproduced the rank and all 120 completions. Its aggregate RMSE differs from batched inference by about `4.1e-14` m.

- [Full benchmark ranking](tail_benchmark_ranking.csv)
- [Per-path ranks](tail_benchmark_by_path.csv)
- [Episode records](tail_benchmark_episodes.csv)
- [15 km/h single-episode inference check](tail_benchmark_15_delay0_ranking.csv)
- [Final checkpoint hashes](final_checkpoints.sha256)

## Independent replay

The same three checkpoints rank first in 8/9 groups on physical seeds 34300–34302 and complete 630/648 runs. The exception is 30 km/h without delay, where MPC ranks first. The 60 km/h, 1000 ms group completes 54/72 episodes. Each other group completes 72/72. The learned coefficients were frozen before this replay.

- [Independent ranking](tail_fresh_ranking.csv)
- [Independent episodes](tail_fresh_episodes.csv)
