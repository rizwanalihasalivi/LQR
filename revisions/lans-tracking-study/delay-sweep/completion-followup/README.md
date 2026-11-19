# Fixed-delay LANS completion follow-up

This experiment starts from the three published refined PPO checkpoints at GitHub commit `676bc50`. It keeps the 1650 kg dynamic bicycle, 100 Hz simulation, full-command FIFO, 10-second horizon, original lateral and heading failure thresholds, four paths, frozen comparators, and failure-penalized RMSE ranking. Candidate changes are confined to the weights of the existing 11-input, 64×64 actor. The refined residual scale and exponential latency coefficient stay at 1.5 and 0.0002 per ms.

## Completion limits

`prefix_bound.py` replays the published initial states and references using the exact scalar simulator until the first physical command can arrive. At 700, 1000, 1500 and 2000 ms, all ten physical slalom cases terminate at steps 66–70 before the first queued steering command can affect the plant. Across three policy seeds this fixes the upper bound at 90/120 per delay. No controller weights can prevent those failures with this queue and termination rule. At 400 ms all 40 physical cases survive the prefix, so the corresponding upper bound is 120/120.

`authority_bound.py` uses sparse linear feasibility to relax the actor's information and allow any bounded steering sequence after the FIFO's forced zero-input prefix. All remaining cases admit a feasible sequence in this relaxation. The LP assumes access to the full reference and individual initial state. It does not establish that a single causal actor can reproduce its actions.

## Refinement and replay

The coefficient fit adjusts only the existing local weights at 400, 1000, 1500 and 2000 ms. The nonlinear fit also uses spare units in the existing second layer at 1000, 1500 and 2000 ms. The robust and rounded variants use fresh physical training and selection seeds near the original reference scales, with a penalty on the worst training scale. The rounded variant recomputes the observation preview with the state-dependent float32 casts. Fits use completion as the first criterion on their selection split and penalized lateral RMSE as the second.

Scalar replay uses the original environment and deterministic PPO inference. The benchmark reuses the published PPO, robust LQR and matched MPC rows. The exploratory independent replay evaluates the candidate, published actor and baselines with common physical seeds 33550–33554. The retained fit incorporates the 400 and 1000 ms changes that survived prototype replay. The final fit also includes the rounded variant's 1500 ms change and keeps the published weights at 2000 ms. These choices use the published benchmark and the exploratory replay. The subsequent final replay on seeds 33650–33654 measures performance on initial states excluded from selection.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python prefix_bound.py
python authority_bound.py
g++ -std=c++17 -O3 -shared -fPIC rollout.cpp -o rollout.so
python tune.py
python replay.py
python replay.py --fresh
python nonlinear_tune.py
python replay.py --variant nonlinear
python replay.py --variant nonlinear --fresh
python tune.py --variant robust
python replay.py --variant robust
python replay.py --variant robust --fresh
python tune.py --variant rounded
python replay.py --variant rounded
python replay.py --variant rounded --fresh
python replay.py --variant retained
python replay.py --variant retained --fresh
python replay.py --variant final
python replay.py --variant final --fresh
```

## Verified final results

The final checkpoint set ranks first at all 11 original sweep levels from 0–2000 ms. It also ranks first at all five independently replayed levels. Each completion count below is out of 120 policy episodes.

| Delay | Published | Final benchmark | Final fresh replay | Benchmark completion upper bound |
| ---: | ---: | ---: | ---: | ---: |
| 400 ms | 117 | 120 | 120 | 120 |
| 700 ms | 90 | 90 | 90 | 90 |
| 1000 ms | 75 | 80 | 90 | 90 |
| 1500 ms | 28 | 31 | 25 | 90 |
| 2000 ms | 17 | 16 | 11 | 90 |

At 2000 ms the published actor also completed 16/120 in this runtime. Its 120 original-benchmark episode records match the final candidate exactly. The historical CSV records 17/120.

- [Final benchmark rankings](final_benchmark_ranking.csv) and [episode records](final_benchmark_episodes.csv)
- [Final fresh rankings](final_fresh_ranking.csv) and [episode records](final_fresh_episodes.csv)
- [Completion bound](prefix_bound_summary.csv) and [actuator feasibility checks](authority_bound_episodes.csv)
