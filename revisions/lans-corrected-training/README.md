# LANS corrected training and precision study

This separate campaign investigates concrete training/evaluation defects while
preserving the diagram: LQR + scalar PPO residual + exponential latency gain +
saturation, eleven observations, the same 64x64 ReLU policy, and the original
dynamic bicycle/full-command FIFO simulator. No predictor, recurrence, queue
input, extra online controller, or per-scenario policy switch is introduced.
Model-informed supervision remains offline; it is not relabelled as a distinct
reinforcement-learning gain.

The older studies, reports and checkpoints remain historical evidence. In
particular, older inspected tests are not fresh tests for this campaign.

## Why the earlier policies did not win every comparison

The earlier LANS ranked first in 14/14 conditions against original baselines,
9/14 against the equally pretrained PPO pool, and 5/14 with objective-matched
MPC. MPC has twenty reference samples against the policy's four. A single
memoryless actor cannot directly observe the varied mass or pending queue state.
LANS and standalone PPO have overlapping control-law expressiveness; the LQR
decomposition does not mathematically guarantee a strictly smaller RMSE.

Review found actual implementation issues, separately from those limitations:

* A 100352-step PPO run used a 200704-step curriculum budget, missing almost all
  exposure to the final mixed/high-delay curriculum stage.
* Projected fine-tuning enforced its bound before rollouts and at the end, not
  after each optimizer update. The corrected code uses an optimizer post-step
  hook and records the number of projected updates.
* Preview conditioning subtracted present state terms after float32 rounding,
  then amplified the lost precision. `PrecisionEnv` computes the algebraically
  identical reference remainder in float64 and converts only at the end.
* Unknown evaluation names silently ran a fallback controller; the runner
  omitted matched MPC; reporting could not handle `rl_ppo` as validation winner.
  Evaluation now validates names and protects existing results/traces.

## Reproduce

Use the sibling `dynamic-bicycle/requirements_snapshot.txt` and
`lans-tracking-study/requirements_extra.txt`. The prepared Python is
`/workspace/lqr-dev/.venv/bin/python`.

From this directory:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
python code/test_precision.py
python code/corpus.py
python code/train.py --kind lans --seed 82
```

Train both `lans` and `ppo` for **all** seeds 82, 83 and 84. The corpus and
model destinations deliberately refuse replacement: use a new campaign to
repeat training. The retained corpus includes failures (379/384 offline teacher
training rollouts complete); those failures are not removed or called successful
deployed LANS episodes. Per-case scores/completion flags are retained in
`results/teacher_cases.json`; the corpus uses causal pre-failure observations,
not post-termination observations. Data augmentation recomputes labels after
perturbation. The historical log key `physical_steering_mse` denotes the
residual-scale-weighted raw action regression loss. It is a training proxy,
not the exact delay-gain-weighted and saturated fused-command MSE.

`results/protocol.json` declares the teacher selection, training seeds, fresh
test cases, metrics and comparison pools before final testing. The teacher grid
in `teacher_search.json` used development cases only. The selected high-gain
candidate improves the development aggregate without claiming universal
dominance. More aggressive gains also produced mixed-delay failures and are
retained in the grid.

Before final evaluation, freeze SHA-256 hashes of all twelve new BC/final
checkpoints and six previous-policy checkpoints in
`results/checkpoint_manifest.json`. Then use `python code/run_eval.py --stage test`
and `python code/report.py --stage test`. All controllers use identical physical
cases. The report retains BC-only comparisons, all three seeds and failures;
the metric is unchanged, including the initial sample and failure penalty.

No full-state/mass oracle is supplied to new neural policies. Old checkpoints
retain their original observation adapter; new policies use the precision fix.
The same plant is used throughout. This is a numerical simulator study, with
small-angle linear-model and finite-case limits, not physical vehicle validation
or a proof of global stability or every-scenario optimality.
