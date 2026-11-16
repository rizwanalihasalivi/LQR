# Same-framework precision and delay-tracking study

This study retains the same dynamic bicycle model/parameter magnitudes,100Hz full-command FIFO execution,LQR+scalar neural PPO residual+exponential gain+saturation topology,and11 original observation channels. It does NOT add a predictor,MPC controller,queue/history input or extra vehicle state to deployed LANS.

Changes are explicit: model-informed offline supervision for the existing policy, invertible conditioning of the four existing40ms preview channels, residual scale1.0rad with final physical steering still clipped±0.5rad, and exponential decay0.002/ms. Thus this is not an unchanged-parameter reproduction. The previous robust LQR gain is retained.

The analytic teacher in code/core.py is an offline training target only. Its performance is NOT LANS performance. code/checks.py disables the teacher function while a saved policy controls the original scalar plant, verifying it is not invoked online. Batch/scalar teacher metrics differ by up to0.2mm due float32 preview rounding amplified by fourth-derivative estimation; completions match. All final learned policies are evaluated solely in the original scalar simulator.

Equal architecture,supervision,corpus and PPO budgets are supplied to a new standalone PPO baseline; the original stronger PPO is retained too. BC-only snapshots and adverse unrestricted fine-tuning results are not hidden.

## Validation-driven corrective trial

The initial6000-update supervised pretraining completed84/84 LANS validation cases. Unrestricted PPO fine-tuning regressed to74/84. Those checkpoints,traces and outcomes are retained. Before any final test,a documented corrective trial freezes actor hidden layers and logstd,and constrains output-layer parameter drift during PPO. This is projected output-layer PPO fine-tuning,not unmodified vanilla PPO. It changes training,not the deployed network/controller topology. Pretraining may explain most gains; an independent PPO-learning benefit must not be asserted without the ablations.

See results/protocol.json,results/corrective_trial.json and full logs.

## Reproduce

Use the sibling dynamic-bicycle requirements snapshot and keep all sibling study directories,which supply unchanged plant,baseline and observation code.

    python code/search.py
    python code/protocol.py
    python code/corpus.py
    python code/run_training.py
    python code/run_eval.py --stage validation
    python code/checks.py
    python code/run_limited.py
    python code/run_eval.py --stage validation --only-limited
    python code/run_eval.py --stage test
    python code/evaluate.py --stage test --kind matched_mpc
    python code/report.py
    python code/make_deliverables.py
    python code/audit.py

Use OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1. Test refuses silent re-execution. Preserve all previous checkpoints,failed episodes and fixed validation decisions. Do not select controllers per scenario. Corpus uses no held-out cases.

Fresh final seeds26000–26004 and reference scales(0.88,1.04),(1.12,0.96) are frozen before network training. Reference families were all in training; this is not unseen-family zero-shot generalization. There is no physical vehicle,nonlinear tire,sensor/actuator hardware,energy or timing superiority validation here. Neither first-place means nor finite episode completion establish universal superiority or global stability.

Install requirements_extra.txt for OSQP1.1.3 used by the supplementary objective-matched MPC. Initial L-BFGS-B reference solver failures are retained in matched_mpc_initial_solver; corrected final QPs are solved with OSQP without changing neural policies or cost. All final controller comparisons and stronger-baseline limitations are in Tracking_Report.md.

Latest final mean-rank result:14/14 original-baseline conditions first,9/14 with equally pretrained PPO,and5/14 with objective-matched MPC. All420 final LANS episodes completed. These finite-case means are not universal superiority or proof of a distinct PPO-learning contribution. Original manuscripts and earlier reports have not been silently overwritten.
