# Fixed-delay CPU steering fast path

This benchmark uses the retained noise-robust LANS and frozen PPO checkpoints, LQR gain and matched MPC solver. It measures repeated single decisions over the same 24 road, speed and delay cells as the preceding CPU study. One pinned logical CPU runs each controller on the same state sequence. Timing covers the decision call through the returned steering command. Controller construction, plant stepping and the transport FIFO take place outside that interval.

The benchmark binds gains, road reference buffers and fixed-delay residual gain once per scenario. A shared CPython bridge reads the live four-state buffer for LQR, PPO and LANS. The actor path evaluates the existing eleven observations, including the live previous and mean delay values. It uses fused multiply-add on structurally sparse first layers with up to four nonzero weights per row and uses Torch's SGEMM for denser layers. This uses the same rule for both actors. A probe to extend the sparse evaluation into the second layer produced 0.00146 rad maximum LANS steering difference. The measured configuration keeps the original SGEMM second and output layers.

The benchmark checks the bound LQR, PPO and LANS commands against the existing Python reference before timing. MPC uses the existing matched solver. It also probes mild and moderate sensor perturbations during parity checks. The replay's state is restored after every perturbation. The bound residual gain applies to this fixed-delay setup. The established general path evaluates the current gain per decision.

## Measured results

Three repeats yielded 56,508 decisions per controller on the Intel Xeon Platinum 8573C.

| Median rank | Controller | Previous median | Fast-path median | Fast-path P95 |
| ---: | --- | ---: | ---: | ---: |
| 1 | LQR | 5.832 μs | 0.779 μs | 1.385 μs |
| 2 | PPO | 12.672 μs | 2.163 μs | 4.847 μs |
| 3 | LANS | 20.515 μs | 3.623 μs | 5.726 μs |
| 4 | MPC | 201.810 μs | 186.921 μs | 364.624 μs |

The LANS median is 5.7× faster than the previous native path. LQR remains first by median, followed by PPO and LANS. LQR evaluates a gain dot product and steering clip. The LANS decision also builds the observations, evaluates the actor, and fuses its residual command.

The maximum LANS steering difference was 5.56 × 10⁻¹⁷ rad over 19,796 clean and 2,474 perturbed inputs. All final timed samples were within 10 ms on this host.

## Reproduce

Run from this directory with the tracking-study dependencies, CPython development headers and the Torch CPU build exporting `sgemm_`.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python bound_timing.py
```

[Timing summary](timing_summary.csv) · [Repeat breakdown](timing_by_repeat.csv) · [Host and parity data](host.json)
