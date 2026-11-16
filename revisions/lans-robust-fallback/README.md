# LANS: tuned robust fallback, unchanged framework

Read [Robust_Fallback_Report.md](Robust_Fallback_Report.md) for measured outcomes, including failures and original stronger PPO references. The target “first in every scenario” is not established.

The original dynamic bicycle plant and parameter magnitudes,11 input channels,64x64ReLU network,LQR+PPO residual+exponential latency gain+saturation,100Hz and full command queue are retained. No queue-history input,observer,predictor or additional controller is introduced. LQR Q/R and policy initialization/training are changed explicitly.

Thirty-three gain candidates are tuned using validation only. All six final policies (three independent LANS seeds,three PPO seeds) have200704steps. The fresh primary test has1260episodes, plus840matched original-checkpoint references. At200ms the original beta suppresses PPO,so high-delay gains are attributable to the new LQR fallback. No neural runtime-control/global safety claim is made.

## Reproduce within this repository

Use the dependencies listed in ../dynamic-bicycle/requirements_snapshot.txt. Keep sibling dynamic-bicycle and lans-improvement directories: unchanged plant/model code and invertible feature scaling are imported from them.

    python code/tune.py
    python code/tune_extended.py
    python code/study_plan.py
    python code/run_training.py
    python code/certificate.py
    python code/run_evaluation.py
    python code/report.py
    python code/run_legacy_reference.py
    python code/legacy_report.py
    python code/prior_reference.py
    python code/make_deliverables.py
    python code/audit.py

Use OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1. The test runner seals test_started.json and refuses silent re-execution. Preserve the original run before reproducing; use a separate study directory or remove generated outputs only in a deliberately new campaign. No final-test-driven retuning or per-scenario controller switching is allowed.

All raw primary and legacy traces are retained. Reports use remaining-horizon failure penalties and completion counts. Ranking means,CI estimates and aggregate RMSE are different summaries; none should be selectively renamed as proof of universal rank1. This is simulation on the supplied linear model,not a real vehicle or nonlinear tire validation. The old manuscript and old results remain unchanged; this report is not a claim that they already incorporate the new study.

A post-hoc no-learning-prior ablation is retained to separate PPO learning from the analytical prior and retuned fallback. It is never used for model selection. Both positive and negative effects are reported.
