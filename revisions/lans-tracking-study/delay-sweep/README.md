# Fixed-delay controller sweep

This study compares robust LQR, equally pretrained PPO, objective-matched MPC and final LANS from the tracking study. It applies each of 11 constant delays (0, 50, 100, 150, 200, 300, 400, 700, 1000, 1500 and 2000 ms) to the full steering command in the dynamic bicycle simulator. Each delay uses four reference paths, two reference scales and five initial-state seeds at 1650 kg. PPO and LANS each use all three final checkpoints (seeds 72–74). The sweep contains 3,520 episodes.

- [Per-delay rankings and RMSE](summary.csv)
- [Per-episode observed and penalized RMSE, completion and configuration](episodes.csv)

Rankings use the mean lateral RMSE with the tracking study's remaining-horizon penalty for incomplete episodes. The overall mean weights each delay equally. Overall order is LANS (1.0775 m), PPO (1.1563 m), LQR (1.4069 m), then MPC (2.1708 m). At 400 ms, LANS completed 7 of 120 episodes. At 700 ms and above, every episode terminated early.

## Reproduce

Use the Python dependencies in the [tracking study setup](../README.md). From this directory, run the sweep in a scratch copy of the repository if you want to preserve the recorded CSVs. The script writes both CSVs next to itself.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python sweep.py
python verify.py
```

Verification compares the 640 episodes at 0 and 200 ms to the tracking study's saved test results.
