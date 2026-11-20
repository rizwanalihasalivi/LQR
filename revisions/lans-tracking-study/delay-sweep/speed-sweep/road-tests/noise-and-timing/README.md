# Sensor noise and controller decision time

The study uses the retained LANS switching checkpoints, frozen limited PPO, the retained LQR gain and objective-matched MPC. The road cases are Straight, Single Lane Change, Double Lane Change and Continuous Curve at 90 and 150 km/h, with 0, 200 and 1000 ms fixed command delay. The plant runs at 100 Hz on the same 250 m roads as the fresh road benchmark. Each noise level covers 24 cells, two road scale pairs, and physical seeds 48200–48204. LANS and PPO each use three policy seeds (72–74). LQR and MPC each run the same ten physical episodes once per cell.

## Sensor noise

The four entries are zero-mean Gaussian measurement noise standard deviations. Draws are independent across ticks and channels. An identical noise stream is used across controllers and checkpoints for each physical seed, and the moderate setting scales the same draws by five.

| Level | Lateral error σ, m | Lateral error rate σ, m/s | Heading error σ, rad | Heading error rate σ, rad/s |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0 | 0 | 0 | 0 |
| Mild | 0.01 | 0.02 | 0.002 | 0.01 |
| Moderate | 0.05 | 0.10 | 0.01 | 0.05 |

LANS, PPO, LQR and MPC use the same noisy state estimate at each tick. The learning policies use their existing preview preprocessing with that estimate. The reference road and speed are known, and the latency inputs retain their benchmark values. The vehicle evolves from the true state and tracking error and completion are scored against it. The clean rows are from the archived fresh benchmark. A zero-noise replay at 90 km/h and 200 ms on Single Lane Change matched the clean episode scores and completions for all four controllers in the validation check.

The cell score is mean failure-penalized lateral RMSE over episodes. The overall score equally weights the 24 cells. A cell can rank first with incomplete episodes, so completion counts are reported alongside rank.

## Decision timing

Each controller receives the same sequence of clean states from one LANS-driven replay per cell, using physical seed 48200, the first road scale pair, and policy seed 72 for both actors. Timing covers batch-one observation construction where required, policy prediction or MPC solve if applicable, and final steering composition and clipping. The timed region ends at the steering command. Model loading, solver construction, simulator stepping, the command FIFO and transport delay are outside it. The benchmark warms each controller, alternates their order by tick, and runs on one pinned logical CPU with one Torch thread. MPC uses the previous decision as its warm start and records whether an OSQP solve was called. This measures the Python implementations on the recorded cloud CPU.

## Reproduce

Run in this folder with the tracking-study dependencies. Run the timing command after the noise replay has finished.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python noise_replay.py
python decision_timing.py
```

## Noise results

| Noise | LANS overall rank | Mean penalized RMSE | LANS cell wins | LANS completed |
| --- | ---: | ---: | ---: | ---: |
| Clean | 1 | 0.420054 m | 22/24 | 625/720 |
| Mild | 1 | 0.390079 m | 15/24 | 638/720 |
| Moderate | 1 | 0.457996 m | 16/24 | 622/720 |

The pooled order is LANS, LQR, PPO, MPC at each noise level. Delay changes the result substantially. At zero delay LANS went from 8/8 wins on clean inputs to 0/8 with either noise level while completing 240/240 episodes in each case. PPO won seven of eight mild-noise cells, with MPC winning the eighth. PPO won all eight moderate-noise cells. The mean zero-delay LANS error rose from 4.30 mm to 9.63 mm and 33.05 mm.

Across the 16 delayed cells, LANS won 15 with mild noise and all 16 with moderate noise. At 1000 ms it won all eight cells under each noise condition, with 145, 158 and 142 completions out of 240 respectively. Mild noise reduced the pooled error in this seeded replay, in part because more 1000 ms episodes completed. The result reflects these noise draws and cases.

[Overall comparison](noise_summary.csv) · [Ranks by road, speed and delay](noise_by_cell.csv) · [Raw episode results](noise_episodes.csv)

## Timing results

Three matched repeats on a single logical CPU of an Intel Xeon Platinum 8573C yielded 54,264 measurements per controller.

| Controller | Median | P95 | P99 | Over 10 ms |
| --- | ---: | ---: | ---: | ---: |
| LQR | 0.0119 ms | 0.0155 ms | 0.0524 ms | 0 |
| PPO | 0.2030 ms | 0.3156 ms | 0.4585 ms | 0 |
| MPC | 0.2132 ms | 0.3930 ms | 0.5276 ms | 0 |
| LANS | 0.2330 ms | 0.3592 ms | 0.5041 ms | 0 |

LQR is fastest. This matched MPC implementation had a lower overall median than LANS and a higher P95. MPC took the unconstrained analytic path in 49.3% of measured decisions, with a 0.122 ms median. When it called the bounded QP solver, its median was 0.280 ms and the P95 was 0.446 ms. The measured set included zero MPC solver fallbacks. A separate pilot pass saw occasional wall-clock samples above 10 ms for LANS, PPO and MPC. The three-repeat table describes observed timing on this host. Embedded hardware and worst-case execution time require separate measurement.

[Timing summary](timing_summary.csv) · [Repeat breakdown](timing_by_repeat.csv) · [Host details](timing_host.json) · [Raw decisions](timing_decisions.csv)
