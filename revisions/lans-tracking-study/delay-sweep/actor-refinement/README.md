# Same-framework LANS actor refinement

The actor uses the tracking study's existing 11 observation channels and two 64-unit ReLU layers. The controller still combines the robust LQR command and the PPO actor residual using exponential latency gain and steering saturation. The network uses the previous measured delay from the existing observation. The complete steering command traverses the original FIFO. The experiment adjusts residual scale to 1.5 and decay to 0.0002 per millisecond. Model-based rollouts optimize the actor initialization offline. The simulator's reference signals are used to build the offline trajectories. Runtime inference uses the original policy observations, and requires no offline teacher or separate solver.

The search uses randomized initial states and reference scales on separate training and validation cases. The offline linear approximation precomputes preview features and omits the scalar environment's intermediate float32 rounding. The checkpoint evaluation uses the unmodified scalar plant, queue and termination conditions. The trained PPO actor receives 8,192 additional sampled training steps for each seed, with hidden layers frozen and a projected output adjustment of about 1e-6. Observed performance predominantly reflects the offline initialization.

The original benchmark uses the published static LQR, PPO and MPC scores. The matched benchmark reruns all four controllers on initial seeds 28710–28714 and reference scale pairs (0.9, 1.025) and (1.1, 0.975). Episode-level CSVs report observed and failure-penalized RMSE and completion. The ranking metric is the average per-episode failure-penalized lateral RMSE.

The matched benchmark now uses the three [frozen Direct Trial checkpoints](../zero-delay-diagnosis/direct-trial/frozen_checkpoints.sha256), one per training seed across all tested delays. The zero-delay refinement uses the existing fourth preview channel alongside the earlier feedback terms. The plant, full-command FIFO, baseline policies and scoring are shared across the actor revisions.

## Results

Direct Trial LANS ranked first at 0, 50, 100 and 1000 ms in the matched benchmark. At 0 ms, its mean penalized RMSE was 0.003381182010 m versus 0.003381223556 m for MPC, a lead of 0.042 micrometers. Both controllers completed every zero-delay episode. At 1000 ms, Direct Trial LANS completed 81 of 120 episodes, compared with 84 of 120 for the earlier actor replayed in the same environment.

The earlier actor ranked first at all 11 original delays and trailed MPC by 0.75 micrometers on the 40 matched zero-delay cases. On the separate 128-case holdout, Direct Trial LANS led MPC by 1.18 micrometers of mean RMSE. See the [zero-delay study](../zero-delay-diagnosis/README.md) for its training and validation protocol.

- [Earlier actor's original sweep ranks](ppo_summary.csv) and [episode data](ppo_episodes.csv)
- [Direct Trial matched ranks](fresh_summary.csv) and [episode data](fresh_episodes.csv)

## Reproduce

Use the installed tracking-study dependencies. To rerun the matched benchmark with the frozen checkpoints, copy the revisions tree to a scratch directory and run from its actor-refinement directory.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python fresh.py all
```

To reproduce the earlier actor's training and original sweep, run these commands in the same directory.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
g++ -std=c++17 -O3 -shared -fPIC rollout.cpp -o rollout.so
python search.py
python nonlinear_zero.py
python refine_actor.py
python extend_actor.py
python finetune.py --extended
python candidate.py --ppo --all
```
