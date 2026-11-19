# Same-framework LANS tuning

The search varied the LQR lateral-error weight, PPO residual scale and exponential delay decay. It evaluated 36 distinct parameter settings on validation seeds 22000 and 22001. One setting was selected across all delays by minimizing the mean log ratio of failure-penalized lateral RMSE to the original LANS. Selection changed the LQR lateral-error weight from 100 to 85. The PPO checkpoints and full-command delay protocol were reused.

The selected setting was rerun in the original scalar simulator on 1,320 LANS benchmark episodes. Rankings use the previously recorded LQR, PPO and MPC results. Tuned LANS ranked first at 150, 200, 300, 400 and 700 ms. At 700 ms it completed zero episodes. Completion increased from 84 to 90 at 300 ms and from 7 to 13 at 400 ms. The rank at 0 ms dropped from second to third.

- [Comparison and ranks](comparison.csv)
- [Selected LANS episode results](selected_episodes.csv)
- [Validation grid](validation.csv)
- [Selected parameters](selected.json)

Use the [tracking study dependencies](../../README.md). Run from this directory in a scratch copy to preserve published results.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python tune.py
python check_scalar.py
```
