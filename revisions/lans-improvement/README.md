# LANS improvement campaign

This is a separate prospective improvement study. The original dynamic-bicycle revision, results and manuscript are unchanged. No universal rank-one claim is made.

The dynamic bicycle model, vehicle parameter magnitudes, 100 Hz ZOH plant, full-command FIFO queue, LQR baseline, scalar PPO residual, 11-input 64x64 actor, exponential latency-gain family and saturation are retained. The study tunes observation scaling, reward/exploration, residual magnitude, gain decay and training coverage. No observer, predictor, MPC branch, oracle or extra state is inserted into LANS.

The frozen machine-readable protocol is results/protocol.json. All three seeds of every configuration are retained. Two LANS configurations and two PPO configurations receive the same step budget and matched training distributions. A single complete configuration per family is selected using validation, never by individual test scenario or best seed. Fresh test initializations and reference parameters are evaluated only after selection. All reference families are included in training; this is not unseen-family zero-shot generalization.

LQR and MPC use the declared previous baseline settings. MPC uses its 20-step reference preview; the policy has the original four preview observations. No optimality across all possible LQR/MPC tunings is claimed. This is not a proof that LANS universally dominates other controllers.

Reproduce with the previous revision's Python dependencies:

    python code/protocol.py
    python code/feasibility.py
    python code/run_training.py
    python code/run_evaluation.py --stage validation
    python code/run_evaluation.py --stage test
    python code/report.py

Training writes separate new checkpoints. Selection refuses to overwrite its locked file. Do not retune on the final test or replace failed seeds. Keep failed episodes and their penalized RMSE.

At 200 ms, the original beta is 1.78e-68: the original bounded PPO residual cannot materially alter the unstable delayed LQR. A second candidate retains the exponential form but tunes its decay to 0.002/ms. This is an openly changed gain parameter, not a demonstrated guarantee.
