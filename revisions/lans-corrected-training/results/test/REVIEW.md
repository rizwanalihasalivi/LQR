# Corrected LANS training: test review

LANS completed 418/420 episodes. The target of first place in every declared comparison is not established.

This is a separate study; historical results and policies were retained. All 2940 current episodes and 2940 raw traces were checked, including failures. Values below are initial-inclusive failure-penalized RMSE in metres, lower is better.

| Comparison pool | LANS first/joint first of 14 | Mean numerical rank |
|---|---|---|
| historical_tracking | 6 | 1.571 |
| equal_pretraining | 9 | 1.357 |
| objective_matched | 5 | 1.786 |
| all_references | 0 | 3.571 |

The historical-tracking pool means the previous tracking PPO plus the original LQR and MPC; it does not relabel the historical policy as a newly trained comparator. Equal-pretraining uses the new corrected PPO; objective-matched additionally replaces the original MPC with matched MPC. All-references includes both BC-only policies and both historical learned families. No per-condition or best-seed switching is used.

| Condition | LANS | New PPO | BC LANS | Previous LANS | Previous PPO | Matched MPC |
|---|---|---|---|---|---|---|
| dlc/fixed200/1650 | 0.069080 | 0.223476 | 0.071422 | 0.037840 | 0.040233 | 2.442537 |
| dlc/mixed/1650 | 0.009321 | 0.007749 | 0.009310 | 0.005609 | 0.005496 | 2.194363 |
| dlc/none/1100 | 0.003904 | 0.004023 | 0.003904 | 0.007133 | 0.007368 | 0.002335 |
| dlc/none/1650 | 0.003792 | 0.003781 | 0.003793 | 0.005155 | 0.005207 | 0.002515 |
| dlc/none/2200 | 0.004179 | 0.004205 | 0.004179 | 0.007601 | 0.007587 | 0.002683 |
| sine/fixed200/1650 | 0.034958 | 0.041369 | 0.036188 | 0.021263 | 0.020928 | 2.442287 |
| sine/mixed/1650 | 0.007836 | 0.007200 | 0.007838 | 0.005285 | 0.005210 | 2.201598 |
| sine/none/1650 | 0.003758 | 0.003766 | 0.003759 | 0.005153 | 0.005188 | 0.002514 |
| slalom/fixed200/1650 | 0.283905 | 0.435101 | 0.285267 | 0.221977 | 0.226315 | 2.442184 |
| slalom/mixed/1650 | 0.159111 | 0.312223 | 0.159057 | 0.014101 | 0.014275 | 2.035451 |
| slalom/none/1650 | 0.003903 | 0.003887 | 0.003904 | 0.005369 | 0.005342 | 0.002514 |
| step/fixed200/1650 | 0.032634 | 0.041159 | 0.033591 | 0.022794 | 0.022813 | 2.442367 |
| step/mixed/1650 | 0.007785 | 0.007230 | 0.007785 | 0.005327 | 0.005242 | 2.215127 |
| step/none/1650 | 0.003765 | 0.003773 | 0.003766 | 0.005157 | 0.005190 | 0.002514 |

## Retained LANS failures

| Train seed | Physical seed | Condition | Amplitude/time scale | Steps | Exit | Final heading (rad) | Max command/applied gap (rad) | Max queue | Completed peer seeds |
|---|---|---|---|---|---|---|---|---|---|
| 83 | 33002 | slalom/mixed/1650 | 1.08/0.94 | 317 | heading limit | -0.504891 | 0.878809 | 13 | 82 |
| 84 | 33002 | slalom/mixed/1650 | 1.08/0.94 | 198 | heading limit | -0.501600 | 0.950314 | 13 | 82 |

The recorded command and applied-command gap reflects the full-command FIFO delay: a newly saturated command can coexist with an older command still being applied. A finite-state heading-limit exit is retained as a control-robustness failure, not a nonfinite-state event. A simulator defect has not been identified by this trace audit. The seed differences are reported rather than selecting only the successful policy.

| Controller | Completed episodes | Aggregate penalized RMSE (m) |
|---|---|---|
| corrected_lans | 418/420 | 0.044852 |
| corrected_ppo | 409/420 | 0.078496 |
| new_bc_lans | 418/420 | 0.045269 |
| new_bc_ppo | 410/420 | 0.081801 |
| previous_lans | 420/420 | 0.026412 |
| previous_ppo | 420/420 | 0.026885 |
| matched_mpc | 61/140 | 1.316499 |
| mpc | 130/140 | 0.294376 |
| fast_lqr | 100/140 | 0.732294 |

## Issues corrected

The inherited 200704-step training curriculum supplied only zero-delay and 0/10ms episodes during the actual 100352-step PPO budget; its mixed and high-delay phases were never reached. Training now covers the declared delay distribution within its actual budget. The output-layer projection previously occurred only before rollouts and after training, allowing intermediate PPO updates outside the declared parameter box; the corrected update hook enforces the constraint after every optimizer step. Both fixes apply to LANS and the equally pretrained PPO.

The preview adapter previously subtracted present-state terms after float32 rounding, amplifying cancellation in derivative-sensitive features. The new adapter computes the algebraically equivalent eleven observations before the final float32 conversion. The offline teacher was selected on the declared development grid; this changes training targets and is not a correction to the physical plant or proof that the deployed controller is an analytic teacher.

Evaluation now rejects unknown controller labels, preflights complete output sets, preserves partial runs, includes matched MPC automatically, and records checkpoint/protocol hashes. Reports include every declared comparator and recompute metrics from raw traces. Historical reporting was also repaired to handle an unrestricted PPO validation winner instead of silently omitting it.

## Checkpoint recommendation

This campaign is experimental. Preserve the stable previous checkpoint; the new policies must not replace it merely because they are newer. On these matched current cases, experimental LANS completed 418/420 with aggregate penalized RMSE 0.044852 m; previous LANS completed 420/420 with 0.026412 m. Completion regressed against the previous checkpoint on matched cases. Aggregate penalized RMSE also regressed against the previous checkpoint. Any future replacement needs an explicit acceptance decision based on the matched comparisons, including failures. All final outcomes are retained; no parameters are retuned after this final test.

## Teacher and training-loss disclosure

The offline teacher completed 379/384 corpus-generation cases; all 5 failures remain in results/teacher_cases.json. Those teacher outcomes are training-data evidence, not deployed LANS results. The corpus includes the recorded prefixes of failed teacher rollouts rather than claiming every teacher case succeeded.

The training log field named physical_steering_mse is an action-scale-weighted output-loss proxy. For LANS it measures the residual prediction error before the latency gain and final steering saturation; it is not measured post-gain physical steering MSE, trajectory RMSE, or deployment performance. Its existing field name is retained to preserve the frozen training records.

## Interpretation and limits

The new BC-only checkpoints are explicit ablations. Differences after constrained PPO updates do not by themselves establish an additional reinforcement-learning benefit. The three training-seed confidence intervals in summary.csv and paired_differences.csv are exploratory and conditional on these physical cases, without multiple-comparison correction. Historical training seeds are not matched random training experiments.

The new policies keep the same 11 causal inputs, 64x64 actor, scalar residual, LQR, latency gain, steering saturation and dynamic bicycle plant. The nominal teacher cannot directly observe unknown vehicle mass. Policy preview is four samples, whereas matched MPC uses twenty. The test is a linear small-angle simulation with no tire, hardware, energy or global-stability validation. Current audit: 0 recorded optimizer failures; passed metric-integrity checks do not turn unsuccessful episodes into successes.
