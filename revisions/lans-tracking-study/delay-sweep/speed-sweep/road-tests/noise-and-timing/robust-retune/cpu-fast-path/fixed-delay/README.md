# Fixed-delay activation specialization and native core timing

The benchmark uses the retained noise-robust LANS and frozen PPO checkpoints, the original LQR gains, and the existing matched MPC solver. It uses the same 24 road, speed and delay cells as the preceding benchmark. All methods receive the same state sequence for each scenario on one pinned CPU core.

For fixed transport delay, the previous and mean delay observations are zero at reset and have the scenario's fixed value after the first command. The two native plans use these phases and the fixed speed to bound the activations of the existing actor. For observations inside a conservative range, a second-layer ReLU whose upper bound is zero cannot contribute. The bound includes a floating-point roundoff allowance. Rows with zero action weight also have no contribution. The packed path uses the same rule for both actors. The LANS plans retain 3–5 possible contributors per phase, while PPO uses the full layer. Out-of-range observations use the dense reference evaluation within the fixed-delay setup.

The specialized path reuses first-layer activations that depend only on the fixed inputs and evaluates the live inputs needed by each plan. It packs the retained second-layer rows into the same SGEMM backend and evaluates the final weighted sum. The Python bridge receives the live state and the tick. Policy weights, observation definitions, residual gain and steering bounds retain their existing values. For varying delays, the previously measured general controller path takes the live previous and mean delay observations.

The steering test compares the specialized outputs with the full Python reference on the common replay states and seeded mild and moderate perturbations. Separate checks exercise the first and subsequent ticks, the dense fallback for out-of-range observations and the invalid-tick guard.

## Timing scopes

The controller timing covers each method's decision call and returned steering command, with the simulation step and transport delay outside the measured interval. The native core diagnosis runs on the same states with the Python call removed for LQR, PPO and both the previous and packed LANS paths. The common C++ clock reads and output store remain in the native sample. MPC uses its existing Python solver entry point in the controller comparison.

## Results

Each comparison includes 56,508 decisions per controller across three repeats on the Intel Xeon Platinum 8573C.

| Median rank | Controller | Decision median | Decision P95 |
| ---: | --- | ---: | ---: |
| 1 | LQR | 0.764 μs | 1.344 μs |
| 2 | PPO | 1.871 μs | 3.913 μs |
| 3 | LANS | 1.913 μs | 3.414 μs |
| 4 | MPC | 192.000 μs | 367.126 μs |

LANS improves its previous 3.623 μs median by 1.89×. It ranks third by median and second by P95. One MPC measurement exceeded 10 ms on this host.

| Native decision core | Median | P95 |
| --- | ---: | ---: |
| LQR core | 28 ns | 32 ns |
| LANS packed core | 168 ns | 204 ns |
| LANS previous core | 526 ns | 555 ns |
| PPO core | 638 ns | 675 ns |

LQR retains the lowest native median after removing Python. LQR performs a gain dot product and steering clip. The LANS core also builds observations and evaluates its residual actor. The packed core has a lower median than the PPO core on the recorded tape.

LANS differed from the Python steering command by at most 5.56 × 10⁻¹⁷ rad across 19,796 clean and 2,474 perturbed checks.

## Reproduce

Run from this directory using the tracking-study dependencies, CPython development headers and the Torch CPU build exporting `sgemm_`.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python fixed_timing.py
```

[Controller timing](timing_summary.csv) · [Native core diagnosis](native_core_timing.csv) · [Repeated measurements](timing_by_repeat.csv) · [Plan sizes](activation_rows.csv) · [Host and parity data](host.json)
