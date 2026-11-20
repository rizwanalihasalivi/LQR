# Nominal recovery at 120 and 150 km/h

The experiment starts from the retained high-speed checkpoints at commit `20d6400`. It targets the two zero-delay cells at 120 and 150 km/h on the original 250 m course. Each cell averages DLC, smooth step, sine and slalom. The plant, steering limit, FIFO, 11 observations, 64×64 actor, benchmark and frozen PPO/LQR/MPC episode data are retained.

The selected variant uses weights in the existing actor to fit two parts of the same steering response. First, an affine reference response minimizes nominal steady-state tracking error. Second, a learned correction acts around the transition between saturated and unsaturated steering. Its features are computed from the observed tracking errors, their derivatives, and the existing preview. The model has no online planning dependency. Legacy units for delays outside the tested 0/200/1000 ms grid provide capacity for these features. The archived checkpoints for those legacy delays stay in place.

Training uses fixed road geometries and disjoint initial-state seeds. The oracle labels come from constrained short-horizon steering solves on training states. Calibration selects checkpoints before the benchmark and independent replays. The frozen benchmark seeds are used for the retrospective prefix diagnostic. Holding the first four commands fixed crosses the MPC target at 120 km/h. Holding the first command fixed crosses it at 150 km/h. The diagnostic allows any legal steering afterwards and omits error beyond update 55.

## Reproduce

Run from this directory with the tracking-study dependencies.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
g++ -std=c++17 -O3 -shared -fPIC recovery.cpp -o rollout_recovery.so
python prefix.py
python recovery.py
python polish.py
python switching.py
python active_set.py
python replay.py --variant switching --only-nominal
python replay.py --variant switching --only-nominal --single-inference
python replay.py --variant switching --split guard
python replay.py --variant switching --split fresh --speeds 120 150 200
```

## Nominal benchmark

Both selected zero-delay groups rank first and complete 120/120 episodes. At 120 km/h the lead over MPC is approximately 0.89 micrometers of average penalized RMSE. At 150 km/h the lead is approximately 0.63 micrometers.

- [Prefix-locked lower bound](locked_prefix.csv)
- [Nominal benchmark](switching_benchmark_nominal_ranking.csv)
- [Episode-level nominal data](switching_benchmark_nominal_episodes.csv)
- [Per-path nominal results](switching_benchmark_nominal_by_path.csv)

## Full grid and independent verification

The same three checkpoints rank first in all 21 original speed-delay groups and complete 2,118/2,520 LANS episodes. The frozen baseline RMSE and completion fields match the prior published speed-sweep data. The two target groups also rank first with single-episode inference. Their measured margins against MPC are 0.883 and 0.632 micrometers at 120 and 150 km/h, respectively.

The independent high-speed replay on seeds 33800–33804 ranks first in 8/9 groups and completes 737/1,080 LANS episodes. At 120 km/h without delay LANS ranks first with 120/120 completions. At 150 km/h without delay it ranks second, with 120/120 completions and an RMSE 0.080 micrometers above MPC. The six high-speed delayed ranks remain first on both splits. At 200 km/h and 1000 ms the original first place corresponds to 1/120 completions.

- [Original 21-group ranking](switching_guard_ranking.csv)
- [Original episode data](switching_guard_episodes.csv)
- [Single-episode inference check](switching_benchmark_nominal_single_ranking.csv)
- [Independent high-speed ranking](switching_fresh_120-150-200_ranking.csv)
- [Checkpoint hashes](final_checkpoints.sha256)
