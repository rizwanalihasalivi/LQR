# Zero-delay LANS trace diagnosis and refinement

The diagnosis replays the previous zero-delay refinement in the scalar simulator against objective-matched MPC. Each episode uses the same initial state and reference for both controllers. The three actor checkpoints are averaged by case. The trace CSV stores lateral squared-error sums for steps 0, 1–10, 11–30, 31–100 and 101–1000, along with sampled early steering commands and saturation counts. The paired phase CSV converts each phase's contribution to micrometers of mean lateral RMSE using the exact paired difference-of-squares identity.

On the old matched cases, most of the net excess error comes in the first ten steps and after step 100. LANS recovers some of the early deficit during steps 11–30. The previous holdout shows the same early and late sources. Both controllers completed all episodes. The slalom exhibits the largest gap.

The trial fits an estimate of the ideal steady tracking state from the existing preview inputs offline. It moves the zero-delay ReLU feedback features toward deviations around that state and optimizes those weights in full-horizon simulations. No extra observations, solver, controller block or lookahead enter deployed LANS. The compiled approximations support offline parameter search. Evaluation uses the scalar environment, full-command delay queue, original success rule and unmodified baselines.

The benchmark replays the prior cases and evaluates seeds 34520–34535, four reference paths, and reference scale pairs (0.925, 1.015) and (1.075, 0.985). The new cases were reserved before tuning and were not used for selection.

A second trial selected zero-delay weights by derivative-free search on separate training and validation cases, with slightly more weight on the first 30 simulation steps. This candidate kept the four preview inputs and changed their coefficients together with the actor's feedback weights. It is recorded in `direct-trial/`. It ranked first on the original, prior matched, and new matched cases by mean failure-penalized lateral RMSE. The leads over objective-matched MPC were about 0.041 and 1.182 micrometers respectively. This is a ranking on the tested simulator cases.

- [Paired diagnostic phases](diagnose_paired_phases.csv)
- [Final original ranks](direct-trial/original_summary.csv)
- [Final prior ranks](direct-trial/prior_summary.csv)
- [Final new matched ranks](direct-trial/new_summary.csv)

## Reproduce

Use the tracking-study Python environment. The prior configuration and checkpoints used by the diagnosis are in `previous/`. Run in a scratch copy to preserve the recorded results.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
g++ -O3 -std=c++17 -fPIC -shared core.cpp -o core.so
g++ -O3 -std=c++17 -fPIC -shared phase.cpp -o phase.so
python traces.py diagnose
python quantify.py
python phase_fit.py
python finetune.py --phase
python evaluate_zero.py prior all
python evaluate_zero.py new all
g++ -O3 -std=c++17 -fPIC -shared forward.cpp -o forward.so
python direct_fit.py
python finetune.py --direct
python evaluate_zero.py original all
python evaluate_zero.py prior all
python evaluate_zero.py new all
```
