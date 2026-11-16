# Response to Reviewer 1: LANS dynamic bicycle revision

This response documents a new revision following the reported rejection; it does not imply an invitation to revise or reviewer/editor acceptance. The LQR–PPO residual framework and dynamic bicycle structure are retained. Vehicle parameter magnitudes are unchanged; stiffness sign convention and sampled LQR design are corrected explicitly. All claimed new experiments have been run on the supplied code and are separable from the archived kinematic results.

## Comment 1: Novelty and neuro-symbolic terminology

**Response and changes.** We agree that LQR plus residual PPO is not itself a new control principle. We retain the LANS control architecture, remove “neuro-symbolic” from the title and scientific claims, acknowledge residual RL [2], and narrow the contribution to an explicitly implemented latency-weighted controller and a reproducible dynamic-model assessment. The exponential gain is labelled prescribed and empirical; no symbolic-reasoning or derived-optimality claim remains.

**Revised manuscript:** I; III-A. **Evidence:** `protocol.json`.

## Comment 2: Role of the latency reward

**Response and changes.** The complete reward and weights are now explicit. We distinguish action-independent exogenous latency costs from policy-dependent tracking/control rewards and from prescribed gain attenuation. A 1,000-transition paired intervention verifies identical state trajectories when only the reward term changes. We separately retrain full and reward-off policies for all three training seeds with matched budgets. Their differences do not establish learned control of runtime; fixed-horizon reasoning and termination/finite-sample caveats are stated.

**Revised manuscript:** III-B/C; VI-A. **Evidence:** `checks.json; paired_differences.csv; training.json files`.

## Comment 3: Actual control execution delay

**Response and changes.** The new simulator delays real steering execution through a timestamped FIFO queue, using ceil(delay/10ms), causal release deadlines and held previous steering. The entire fused command is delayed in the main results. Queue boundary checks and every per-step generated/applied command and generation timestamp are supplied. Policy inputs use previous delays, preventing future-delay leakage. Synthetic transport delay is not presented as physical CPU blocking.

**Revised manuscript:** III-D; IV; VI-B. **Evidence:** `checks.json; traces/*.npz; dynamic_env.py`.

## Comment 4: 15 ms LQR margin versus 200 ms fallback claim

**Response and changes.** We withdraw the arbitrary-delay and >200ms stability guarantee. On the same dynamic bicycle model with a correctly sampled LQR, fixed total-command delays of 20–300ms are not Schur stable in the tested scan. Disabling PPO cannot remove delay on LQR. The full-command and residual-only channels are reported separately; the latter is a diagnostic rather than a replacement architecture. Original conflicting 15/50ms limits are removed.

**Revised manuscript:** II-B; V; VI-B; Fig.1. **Evidence:** `model_audit.json; fixed_delay_stability.json; episodes_all.csv`.

## Comment 5: Fair 49-fold speed comparison

**Response and changes.** Every method is now timed on the same cloud CPU, states, single-thread/batch setting and Python decision boundary, with warmup and random interleaving. Measured PPO latency replaces the 100Hz simulation-period proxy. Mean/P99/observed maximum and all 21,000 samples are supplied. The original 49-fold speedup claim is withdrawn, and the fixed100Hz control rate is distinguished from reciprocal computation throughput.

**Revised manuscript:** VII; Table6. **Evidence:** `timing_raw.csv; timing_summary.json; runtime.json`.

## Comment 6: Implausible timing ablations

**Response and changes.** Reward-on/off training uses the same network dimensions; no-gain and no-LQR deployment interventions share the full trained checkpoint. Their timing is measured under the same procedure and no artificial 100Hz timing floor is imposed. Retraining ablations are distinguished from frozen-policy interventions. The original 4930-to100Hz ablation table and causal speedup interpretation are replaced.

**Revised manuscript:** III-B; IV; VI-A; VII. **Evidence:** `timing_raw.csv; episodes_all.csv; train.py`.

## Comment 7: Worse nominal tracking than LQR/MPC

**Response and changes.** We explicitly report nominal LQR, matched MPC, full residual controller, frozen no-LQR/no-gain interventions, and separately trained PPO. The residual-off full architecture equals the LQR baseline. Paired RMSE differences, residual RMS, saturation and per-step steering/error traces quantify when the residual worsens the baseline. We do not claim PPO must improve nominal linear LQR, and discuss objective mismatch and finite training without asserting an untested universal cause.

**Revised manuscript:** VI-A; Tables1/2; Fig.2. **Evidence:** `summary.csv; paired_differences.csv; traces/*.npz`.

## Comment 8: Overstated generalization

**Response and changes.** DLC is identified as the training shape and cannot establish out-of-distribution generalization. Step, sine and slalom are evaluated without retuning, on all three policies and five common physical initializations. All failures are retained and penalized RMSE is explicit. Generalization is restricted to the tested cases; large errors and the linear-model validity boundary are not hidden.

**Revised manuscript:** IV; VI-C; Table4. **Evidence:** `summary.csv; episodes_all.csv`.

## Comment 9: Incomplete stability proof

**Response and changes.** The global/arbitrary-delay proof is replaced with a discrete Lyapunov identity and a numerical local invariant unsaturated ellipsoid for the zero-reference zero-residual LQR. A bounded-forcing inequality gives an ultimate-bound interpretation, not convergence to zero. Nonzero residual equilibrium, saturation dependence and fixed-delay augmented dynamics are treated explicitly. No unsupported general time-varying-delay or global saturated-hybrid theorem remains.

**Revised manuscript:** V; Fig.1. **Evidence:** `checks.json; fixed_delay_stability.json; model_audit.py`.

## Comment 10: Missing reward, policy, window, baseline and seed details

**Response and changes.** We specify all reward coefficients, Gaussian mean/log_std implementation and clipping,11 inputs and scaling, ten-sample causal window, exact gain units, queue semantics, PPO hyperparameters, horizon/termination, MPC horizon/solver/tolerances, and model parameters. Nine fresh final checkpoints each have200704steps, across three independent training seeds per family. Evaluation seeds are separately identified; confidence intervals use training-seed means rather than inflating n with repeated episodes. The new matched budget is not claimed as historical training replication.

**Revised manuscript:** II–IV; Tables1–5. **Evidence:** `protocol.json; source; models/*/training.json; Monitor logs`.

## Comment 11: Unfair energy and blind-distance comparison

**Response and changes.** We remove the GPU/CPU energy ratio,305-fold/1-300 claims and deployment-safety interpretation. This host exposes no workload energy counter, so no new power measurement is invented. Computation-only travel distance is recomputed from matched measured decision times and explicitly separated from unmeasured sensor-to-actuator delay. Energy verification is outside the remaining claims.

**Revised manuscript:** VII. **Evidence:** `runtime.json; timing_summary.json`.

## Comment 12: Numerical and reference inconsistencies

**Response and changes.** All scientific result tables and figures are generated directly from fresh CSV/JSON files. The original0.199/49x/305x/15-vs50ms/18-vs50x and AblationC inconsistencies are not reused. Model signs,continuous-versus-sampled LQR and gain continuity are corrected explicitly. The bibliography is replaced by four relevant sources with metadata cross-checked against accessible official project sources; direct publisher queries were network-blocked. The unverified repository-completeness statement is removed. Authors should still perform final venue-specific bibliography/style review before submission.

**Revised manuscript:** II; V–VIII; References. **Evidence:** `model_audit.json; source snapshots; integrity_manifest.json; audit.json`.

## What is and is not resolved

All twelve concerns receive a concrete correction, new evidence, or removal/narrowing of the unsupported claim. This does not mean twelve positive performance findings or guaranteed acceptance. Global stability under arbitrary bounded delay, physical vehicle validation, energy superiority and certified real-time deployment are not established and are no longer claimed. The new campaign is finite-budget reconstruction on the requested dynamic model, not a recovery of missing historical code/results.
