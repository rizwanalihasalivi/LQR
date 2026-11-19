# Same-framework LANS actor refinement

The actor uses the tracking study's existing 11 observation channels and two 64-unit ReLU layers. The controller still combines the robust LQR command and the PPO actor residual using exponential latency gain and steering saturation. The network uses the previous measured delay from the existing observation. The complete steering command traverses the original FIFO. The experiment adjusts residual scale to 1.5 and decay to 0.0002 per millisecond. Model-based rollouts optimize the actor initialization offline. The simulator's reference signals are used to build the offline trajectories. Runtime inference uses the original policy observations, and requires no offline teacher or separate solver.

The search uses randomized initial states and reference scales on separate training and validation cases. The offline linear approximation precomputes preview features and omits the scalar environment's intermediate float32 rounding. The checkpoint evaluation uses the unmodified scalar plant, queue and termination conditions. The trained PPO actor receives 8,192 additional sampled training steps for each seed, with hidden layers frozen and a projected output adjustment of about 1e-6. Observed performance predominantly reflects the offline initialization.

The original benchmark uses the published static LQR, PPO and MPC scores. The additional benchmark reruns all four controllers on new matched initial seeds 28710–28714 and new reference scale pairs (0.9, 1.025) and (1.1, 0.975). Its cases were not used for training or parameter selection. Episode-level CSVs report observed and failure-penalized RMSE and completion. The ranking metric is the average per-episode failure-penalized lateral RMSE.

## Results

LANS ranked first at all 11 original delays. On the additional matched cases, LANS ranked first at 50, 100 and 1000 ms, and second to MPC at 0 ms. Both zero-delay comparisons were decided by less than 1 micrometer of mean RMSE. The study measures simulation performance.

- [Original sweep ranks](ppo_summary.csv) and [episode data](ppo_episodes.csv)
- [Additional matched ranks](fresh_summary.csv) and [episode data](fresh_episodes.csv)

## Reproduce

Use the installed tracking-study dependencies and keep this directory inside the repository. Run in this experiment directory.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
g++ -std=c++17 -O3 -shared -fPIC rollout.cpp -o rollout.so
python search.py
python nonlinear_zero.py
python refine_actor.py
python extend_actor.py
python finetune.py --extended
python candidate.py --ppo --all
python fresh.py all
```
