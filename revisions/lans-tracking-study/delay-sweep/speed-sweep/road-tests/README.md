# Dedicated road tests

The test uses the existing 1650 kg dynamic bicycle model, 250 m forward course, 10 ms step and full-command FIFO at 0, 200 and 1000 ms. Each road is evaluated at 15, 30, 60, 90, 120, 150 and 200 km/h. The existing steering bound, failure thresholds, eleven observations and 64×64 LANS actor are retained. PPO is the frozen limited PPO baseline and MPC uses the matched tracking/control objective.

| Road | Reference |
| --- | --- |
| Straight | Zero lateral offset and derivatives over the full course |
| Single Lane Change | Existing smooth step with a nominal 3.5 m shift |
| Double Lane Change | Existing smooth outbound and return lane change |
| Continuous curve | Smooth 30 m entry into constant spatial second derivative, with nominal end tangent about 11° from straight |

The curve is represented as a shallow forward road in the existing small-angle coordinates. A true 180° turnaround requires a road-relative model that supports reversal of global heading and corresponding longitudinal progress.

## Metric

Each speed-delay-road cell ranks controllers by the mean failure-penalized lateral RMSE of their episodes. The road-wide rank equally weights the 21 cell scores for that road. Completion is reported separately. LANS and PPO each use three policy seeds and the same two amplitude/time perturbations and five physical seeds. LQR and MPC each use the same ten physical episodes per cell. Amplitude and time perturbations leave the straight reference at zero.

The original lane-change results are read from the retained replay. The straight and curve results use the same physical seeds as those results. The fresh replay runs all four roads from seeds 48200 through 48204. Both use the retained switching checkpoints with policy seeds 72 through 74.

## Original road-wide result

| Road | Winner | LANS mean RMSE | Runner-up mean RMSE | LANS cell wins | LANS completed |
| --- | --- | ---: | ---: | ---: | ---: |
| Straight | LANS | 0.250725 m | LQR · 0.595567 m | 15/21 | 570/630 |
| Single Lane Change | LANS | 0.336878 m | LQR · 0.832974 m | 19/21 | 552/630 |
| Double Lane Change | LANS | 0.411866 m | LQR · 1.017332 m | 19/21 | 540/630 |
| Continuous curve | LANS | 0.330607 m | LQR · 0.705112 m | 18/21 | 548/630 |

The 90 km/h straight loss at 200 ms is visible early. On original seeds 26000–26004, scale (0.88, 1.04), LANS policy 72 issued a 0.5 rad magnitude command on the first tick on all five episodes. The mean of the five episode RMSEs over the next 160 samples was 53.5 mm for LANS and 22.7 mm for LQR. LQR's mean first steering magnitude was 0.00352 rad. The FIFO holds each first command for the same 200 ms.

At 200 km/h and zero delay, MPC led all four retained road cases by about 0.078–0.094 mm. At 150 km/h and 1000 ms, the retained DLC controllers each had zero completions. The existing full-preview actuator feasibility relaxation classified all ten original DLC initial conditions as feasible at this point. That indicates scope for a different causal policy within the steering bound.

[Per-road scores](original_by_road.csv) · [Per-cell ranks](original_ranking.csv) · [New-road episodes](original_episodes.csv)

## Reproduce

Run here with the tracking-study dependencies.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python benchmark.py
python benchmark.py --fresh
python summarize.py original_ranking.csv fresh_ranking.csv
python tune.py
python benchmark.py --tuned --lans-only
python summarize.py tuned_original_ranking.csv
python tune.py --extended
python benchmark.py --tuned extended --lans-only --fit-cells
python benchmark.py --fresh --tuned extended --fit-cells --speeds 30 200
```

The tuning probe fits existing residual actor weights at 90 km/h and 200 ms, 150 km/h and 1000 ms, and 200 km/h and 0 ms using physical seeds and amplitude/time perturbations separate from both reported benchmark splits. The probe uses the repository's compiled rollout for search and the original Python simulator and deterministic policy inference for scoring. The retained checkpoints and the replay they came from are kept separately.

The extended probe targets 15 km/h at 1000 ms, 30 and 60 km/h at 200 ms, and 200 km/h at 1000 ms.

## Tuning outcomes

The three-cell probe retained 71/84 individual first places on the original grid. At 200 km/h and zero delay, it reduced LANS error on all four roads and MPC retained first place in each. At 150 km/h and 1000 ms, DLC moved from #3 to #1 with 0/30 completions. At the same speed and delay, Single Lane Change moved from #1 to #2 and from 6/30 to 0/30 completions. That trade-off favors retaining the switching checkpoints for the four-road result.

The extended probe at 30 km/h and 200 ms moved Straight from #2 to #1 at 30/30 completions. The penalized RMSE fell from 4.895 mm to 3.816 mm, ahead of LQR's 4.882 mm. The other three roads kept their first places in that cell. At 200 km/h and 1000 ms, straight-road completions rose from 0/30 to 12/30 and LQR kept first place. Curve improved from #3 to #2 with 0/30 completions.

[Three-cell probe, full original grid](tuned_original_ranking.csv) · [Extended probe, original target cells](extended_original_fit_cells_ranking.csv)

## Fresh replay

The retained checkpoint set wins all four road-wide rankings on physical seeds 48200–48204. LANS wins 75/84 individual cells and completes 2,245/2,520 episodes.

| Road | Winner | LANS mean RMSE | Runner-up mean RMSE | LANS cell wins | LANS completed |
| --- | --- | ---: | ---: | ---: | ---: |
| Straight | LANS | 0.257046 m | LQR · 0.640434 m | 16/21 | 568/630 |
| Single Lane Change | LANS | 0.282405 m | LQR · 0.837664 m | 20/21 | 572/630 |
| Double Lane Change | LANS | 0.398082 m | LQR · 1.015345 m | 20/21 | 540/630 |
| Continuous curve | LANS | 0.290621 m | LQR · 0.701275 m | 19/21 | 565/630 |

The extended probe wins all four roads on the same fresh seeds at 30 km/h and 200 ms, each with 30/30 completions. Its Straight RMSE there falls from the retained policy's 8.603 mm to 7.201 mm. At 200 km/h and 1000 ms, the extended probe moves Straight from #2 and 0/30 to #1 and 12/30. The other three fresh road rankings at that condition stay #1, with 6/30 Single Lane Change completions and 0/30 each for DLC and Curve. The extended fit was replayed on its original target cells and on these two fresh conditions.

[Fresh per-road scores](fresh_by_road.csv) · [Fresh per-cell ranks](fresh_ranking.csv) · [Fresh episodes](fresh_episodes.csv) · [Fresh extended probe](extended_fresh_30-200_fit_cells_ranking.csv)
