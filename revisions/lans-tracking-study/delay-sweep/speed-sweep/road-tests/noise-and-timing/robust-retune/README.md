# Noise-robust LANS weights and CPU decision timing

The same LANS residual architecture uses eleven observations, its 64×64 ReLU actor, and its existing LQR term, residual gain, steering bound and command FIFO. This experiment fits weights in the actor rows used for zero-delay operation at 90 and 150 km/h. The actor receives its existing observations at inference. Both tested noise levels and clean input use the same checkpoint. The source switching checkpoints and the earlier measurements remain available.

## Tracking experiment

The fit uses separate physical seeds 55100–55104 and validates on seeds 56300–56307, with road scales and Gaussian draws separate from the comparison replays. Both fits use all three noise settings in training. The first fit gives the three settings similar weight. The selected `moderate` fit weights moderate-noise tracking more heavily. Both change the nominal feedback and preview weights. The first fit reached 24/24 on mild noise and 16/24 on moderate noise on the paired seeds. The selected fit was then replayed on independent physical seeds.

The paired comparison uses the original physical seeds 48200–48204 and frozen PPO, LQR and MPC episodes from the published [sensor noise study](../README.md). The independent comparison uses seeds 70200–70204 and reruns every controller. The two benchmark road scales, Gaussian noise standard deviations, four road types and three transport delays follow that study. Ranking uses mean failure-penalized lateral RMSE.

| Noise level | Retained wins on paired seeds | Selected wins on paired seeds | Selected wins on independent seeds | Selected completions, paired | Selected completions, independent |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 22/24 | 16/24 | 16/24 | 645/720 | 632/720 |
| Mild | 15/24 | 24/24 | 24/24 | 636/720 | 632/720 |
| Moderate | 16/24 | 24/24 | 24/24 | 617/720 | 628/720 |

The selected LANS checkpoint ranks first overall at each noise level on both seed sets. It wins all 48 noisy cells on each set, including all 16 noisy zero-delay cells. With clean inputs MPC wins the eight zero-delay cells, where LANS ranks second. LANS wins all 16 clean delayed cells. On the paired seeds, LANS completions change from 625 to 645 in clean conditions, from 638 to 636 with mild noise and from 622 to 617 with moderate noise. The rank improvements and the completion changes measure different outcomes.

[Paired ranks](moderate_paired_by_cell.csv) · [Independent ranks](moderate_independent_by_cell.csv) · [Independent summary](moderate_independent_summary.csv) · [First fit](noise_paired_summary.csv)

## CPU timing setup

The native entry point computes the existing observation transform, actor and steering command in compiled C++. Both neural controllers use the same float32 matrix multiplication exported by the installed Torch CPU build. LQR uses the native gain dot product and steering bound. MPC uses the benchmark's matched solver and composition. The CPU comparison uses the same clean states, initial conditions and road references for each controller. As in the earlier timing study, it measures per-decision wall time and excludes vehicle stepping and transport delay.

The preliminary hand-written matrix evaluation differed from the reference steering command by up to 0.00146 rad for LANS because of float32 accumulation around large gate coefficients. The final build uses the same SGEMM operation as Torch and verifies numerical agreement with the Python controller before timing. This experiment reports the native decision path separately from the Python replay used to score tracking.

## Reproduce

Run in this folder with the tracking-study dependencies, a C++17 compiler, and the Linux Torch CPU build exporting `sgemm_`. Run timing after the replay processes finish.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python fit_nominal.py
python verify.py
python fit_nominal.py --variant moderate
python verify.py --variant moderate
python verify.py --variant moderate --independent
python native_timing.py --variant moderate
```

## CPU timing results

Three native repeats on a single logical CPU of the Intel Xeon Platinum 8573C yielded 56,508 measurements per controller. The matched eager pass used the selected actor and identical states. The maximum absolute LANS steering difference was 5.56 × 10⁻¹⁷ rad across 19,796 parity checks, including warm-up decisions. PPO's maximum difference was 5.97 × 10⁻⁸ rad.

| Controller | Matched eager median | Native comparison median | Native comparison P95 |
| --- | ---: | ---: | ---: |
| LQR | 0.0113 ms | 0.0058 ms | 0.0163 ms |
| PPO | 0.2013 ms | 0.0127 ms | 0.0309 ms |
| LANS | 0.2262 ms | 0.0205 ms | 0.0355 ms |
| MPC | 0.1979 ms | 0.2018 ms | 0.3876 ms |

The LANS median fell by 11.0× with native execution. LANS was faster than the matched MPC implementation in this comparison. LQR and PPO had lower medians. LQR computes a gain dot product and clips the output. LANS also prepares observations, evaluates the actor and fuses its output. All the final measured decisions stayed within 10 ms. A preceding native run saw three MPC decisions exceed 10 ms. These shared-host measurements describe this workload and implementation.

[Native timing](native_timing_summary.csv) · [Per-repeat timing](native_timing_by_repeat.csv) · [Matched eager timing](matched_eager_timing_summary.csv) · [Host and parity results](native_timing_host.json)
