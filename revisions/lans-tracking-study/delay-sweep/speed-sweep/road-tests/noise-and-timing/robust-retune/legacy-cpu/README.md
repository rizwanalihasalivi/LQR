# Legacy Intel CPU timing

This is a standalone benchmark for a 64-bit Intel-compatible Linux system or Intel Mac. The source bundle uses baseline x86-64 instructions with AVX disabled. The benchmark requires a C99 compiler and a C++17 compiler on the target. The bundle includes controller weights, a common state tape and the generated OSQP solver sources. Model inference and timing build from this source without a Python runtime.

## Run on the physical machine

Extract `lans-legacy-intel.tar.gz`, open a terminal in the extracted folder, and run the following commands.

```sh
sh build.sh
sh run.sh
```

The run writes the actual hardware model, OS and compiler to `host.txt`. It writes rank, median, P95 and maximum steering difference in `timing.csv`. It pins one CPU on Linux. macOS follows the OS scheduler. Keep the same run count and power settings when comparing the older machines.

## Workload and scope

The tape contains 18,836 common states across the existing 24 road, speed and fixed-delay cells. It uses the same retained LANS weights and the same PPO weights, LQR gains and matched MPC quadratic objective as the Xeon study. The program measures three repeats by default, rotating controller order each tick. The timer covers steering computation and excludes tape loading, scenario setup, simulation stepping and transport delay.

The portable native backend uses scalar float32 actor evaluation and OSQP's generated bounded solver. This changes the implementation under test compared with the prior Torch/MKL and Python-solver benchmarks. Use this bundle on each machine for cross-hardware comparisons. The portable runner checks the command against the stored reference at every state, with limits of 0.003 rad for LANS, 0.00001 rad for PPO, 1e-12 rad for LQR and 0.00003 rad for MPC. LANS's gate sums are sensitive to the float32 reduction order. The generated MPC solver implements the same objective and bounds with embedded solver numerics. The differences and checks cover the recorded workload.

## Regenerate

From this source directory in the tracking-study environment, run `python export_bundle.py /path/to/new-bundle` with the usual single-threaded numerical environment. It exports the reference tape and generates two OSQP solver workspaces. The bundle carries the OSQP source license.

## Verification

The portable build completed 56,508 timed decisions per controller on the Intel Xeon Platinum 8573C cloud host. The Linux executable reports the baseline x86-64 ISA. In QEMU user-mode, the `Conroe-v1` Core 2 profile and the `IvyBridge-v1` profile each completed 18,836 commands per controller. The recorded maximum steering differences from the earlier reference were 0.001465 rad for LANS, 0.000001058 rad for PPO, about 2.78e-17 rad for LQR and 0.00001988 rad for MPC. The emulation checks record command counts and errors.

Physical Core 2 Duo and 2012 MacBook measurements require access to those computers. The Ivy Bridge Linux emulation verifies the instruction profile. The macOS build and actual device performance remain to be measured on the Mac.

[Source bundle](lans-legacy-intel.tar.gz) · [Xeon smoke timing](xeon-portable-smoke.csv) · [Xeon host](xeon-host.txt) · [Emulated checks](emulation-checks.csv)
