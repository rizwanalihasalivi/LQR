# Prospective LANS improvement study: measured outcome

The requested target was LANS first in every scenario. **The final test places the selected LANS first in 0/14 conditions.** A first-place mean in a finite test is not universal or statistically established dominance. All failures and all trained seeds are retained.

## Preserved framework and changed training

The same dynamic bicycle plant, physical parameter magnitudes, sampled LQR gain, full-command queue, LQR + PPO + exponential latency gain + saturation, 11 causal features and 64x64 ReLU networks are retained. No predictor, observer, planner, oracle or extra vehicle-state input is added to LANS. Observation scaling is invertible; preview curvature information is exposed from the existing four inputs. Candidate reward/exploration, residual scale, gain decay and training distribution changes are explicit in the frozen protocol.

There are 12 new policies: 4 configurations × 3 independent seeds, each with 200,704 steps (2,408,448 steps total). PPO receives the same two-configuration training-search budget. This is a finite-budget study, not evidence of converged optimal policies.

Validation uses seeds8000–8001 and nominal reference parameters. A single whole configuration per family is selected by completion rate, then penalized RMSE, across all seeds/conditions. It is never switched by scenario. Selected LANS: **precision**; selected PPO: **ppo_standard**.

The held-out test uses five new initialization seeds12000–12004 and two held-out amplitude/time-scale combinations(0.9,1.1),(1.1,0.9). These reference families were included in training: this is new-parameter testing, not unseen-family zero-shot generalization. There are1,120 final test episodes.

## Overall outcome

| Controller configuration | Mean rank | First-place conditions | Completed / episodes |
|---|---|---|---|
| mpc | 1.143 | 13/14 | 130/140 |
| lqr | 2.571 | 0/14 | 100/140 |
| precision | 2.643 | 0/14 | 300/420 |
| ppo_standard | 3.357 | 1/14 | 280/420 |

Every condition is equally weighted in mean rank. Rank is based on mean failure-penalized RMSE, with ties sharing a rank. A failed case can receive a numeric rank from the declared failure penalty; that rank does not mean successful tracking. Completion counts are therefore displayed.

## Each condition

| Condition | Results: rank, penalized RMSE, completion |
|---|---|
| dlc / fixed200 / 1650kg | MPC rank 1 (0.4715m, 10/10); PPO rank 2 (2.1653m, 0/30); LANS rank 3 (2.3896m, 0/30); LQR rank 3 (2.3896m, 0/10) |
| dlc / mixed / 1650kg | PPO rank 1 (0.0441m, 30/30); LQR rank 2 (0.0450m, 10/10); MPC rank 3 (0.0464m, 10/10); LANS rank 4 (0.0478m, 30/30) |
| dlc / none / 1100kg | MPC rank 1 (0.0099m, 10/10); LANS rank 2 (0.0156m, 30/30); LQR rank 3 (0.0156m, 10/10); PPO rank 4 (0.0412m, 30/30) |
| dlc / none / 1650kg | MPC rank 1 (0.0203m, 10/10); LANS rank 2 (0.0280m, 30/30); LQR rank 3 (0.0281m, 10/10); PPO rank 4 (0.0429m, 30/30) |
| dlc / none / 2200kg | MPC rank 1 (0.0368m, 10/10); LANS rank 2 (0.0424m, 30/30); LQR rank 3 (0.0424m, 10/10); PPO rank 4 (0.0472m, 30/30) |
| sine / fixed200 / 1650kg | MPC rank 1 (0.3476m, 10/10); PPO rank 2 (2.1591m, 0/30); LANS rank 3 (2.3973m, 0/30); LQR rank 3 (2.3973m, 0/10) |
| sine / mixed / 1650kg | MPC rank 1 (0.0092m, 10/10); LQR rank 2 (0.0151m, 10/10); LANS rank 3 (0.0178m, 30/30); PPO rank 4 (0.0399m, 30/30) |
| sine / none / 1650kg | MPC rank 1 (0.0073m, 10/10); LANS rank 2 (0.0148m, 30/30); LQR rank 3 (0.0148m, 10/10); PPO rank 4 (0.0394m, 30/30) |
| slalom / fixed200 / 1650kg | MPC rank 1 (2.2684m, 0/10); LANS rank 2 (2.3820m, 0/30); LQR rank 2 (2.3820m, 0/10); PPO rank 4 (2.3838m, 0/30) |
| slalom / mixed / 1650kg | MPC rank 1 (0.2589m, 10/10); LQR rank 2 (0.2656m, 10/10); LANS rank 3 (0.2665m, 30/30); PPO rank 4 (1.1435m, 20/30) |
| slalom / none / 1650kg | MPC rank 1 (0.1806m, 10/10); LQR rank 2 (0.2160m, 10/10); LANS rank 3 (0.2162m, 30/30); PPO rank 4 (1.0818m, 20/30) |
| step / fixed200 / 1650kg | MPC rank 1 (0.3819m, 10/10); PPO rank 2 (2.1729m, 0/30); LANS rank 3 (2.3905m, 0/30); LQR rank 3 (2.3905m, 0/10) |
| step / mixed / 1650kg | MPC rank 1 (0.0109m, 10/10); LQR rank 2 (0.0150m, 10/10); LANS rank 3 (0.0208m, 30/30); PPO rank 4 (0.0402m, 30/30) |
| step / none / 1650kg | MPC rank 1 (0.0104m, 10/10); LANS rank 2 (0.0143m, 30/30); LQR rank 3 (0.0144m, 10/10); PPO rank 4 (0.0397m, 30/30) |

## Limits and interpretation

The old and new campaigns use different test reference parameters, seeds and training coverage. Do not treat a lower number here than in the old paper as a matched improvement estimate. New LANS and new PPO/LQR/MPC are compared within this test. The LQR and MPC settings are the previously declared fixed configurations; the results do not establish superiority over all possible baseline tuning. MPC has20-step preview while policy inputs retain four previews.

At200ms the original exponential gain is1.78e-68. Bounded PPO actions therefore cannot materially repair the original delayed LQR, and opposite extreme actions produce identical simulated trajectories in the diagnostic. The tolerant candidate openly changes the decay parameter while retaining the same exponential family. This does not establish arbitrary-delay stability.

Three independent training seeds give wide uncertainty. Seed-level paired confidence intervals are supplied in paired_differences.csv; a numerical rank alone does not establish statistical superiority. No energy, compute-speed, physical vehicle safety, nonlinear dynamics or global stability superiority is claimed.

Original manuscript/results/checkpoints remain unchanged. This experimental report must not be described as a successful proof of the requested target unless the actual result supports it.

## Matched checks against the earlier stronger checkpoints

The original full-LANS and PPO policies (seeds42–44) were also run on exactly these new test cases, using their exact original observation API. There are840 supplementary episodes. These checks do not change the locked validation selection; they prevent attributing a rank change to a weaker retrained PPO baseline.

| Condition | New LANS RMSE | Earlier LANS RMSE | Earlier PPO RMSE |
|---|---:|---:|---:|
| dlc / fixed200 / 1650kg | 2.3896 | 2.3896 | 2.3362 |
| dlc / mixed / 1650kg | 0.0478 | 0.0653 | 0.0191 |
| dlc / none / 1100kg | 0.0156 | 0.0192 | 0.0126 |
| dlc / none / 1650kg | 0.0280 | 0.0287 | 0.0172 |
| dlc / none / 2200kg | 0.0424 | 0.0406 | 0.0229 |
| sine / fixed200 / 1650kg | 2.3973 | 2.3973 | 2.3446 |
| sine / mixed / 1650kg | 0.0178 | 0.0587 | 0.0137 |
| sine / none / 1650kg | 0.0148 | 0.0182 | 0.0108 |
| slalom / fixed200 / 1650kg | 2.3820 | 2.3820 | 2.4370 |
| slalom / mixed / 1650kg | 0.2665 | 0.2557 | 0.1523 |
| slalom / none / 1650kg | 0.2162 | 0.2005 | 0.1503 |
| step / fixed200 / 1650kg | 2.3905 | 2.3905 | 2.3391 |
| step / mixed / 1650kg | 0.0208 | 0.0598 | 0.0135 |
| step / none / 1650kg | 0.0143 | 0.0181 | 0.0108 |

A supplementary four-controller comparison retaining the stronger earlier PPO yields these mean condition ranks:

- mpc: 1.429
- legacy_ppo: 1.786
- lqr: 3.214
- new_lans: 3.286

This supplementary ranking is explicitly separate from the prospectively selected comparison. The target of LANS first in every condition remains unachieved. New-versus-old means do not establish statistically significant improvement; the policies use different training seeds and training distributions.
