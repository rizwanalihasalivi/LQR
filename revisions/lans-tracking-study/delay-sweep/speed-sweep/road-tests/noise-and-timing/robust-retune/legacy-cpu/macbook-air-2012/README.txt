LANS benchmark for the Intel MacBook Air (Mid 2012)

1. Extract the entire ZIP into a folder you can write to.
2. Open the extracted folder and double-click Run-Benchmark.command.
3. After the result table appears, find timing.csv and host.txt in the folder.

From a Terminal already in the extracted folder, you can also run
    sh Run-Benchmark.command

The included executable targets Intel x86-64 macOS 10.13 High Sierra
or later, including macOS 10.15 Catalina. It includes the solver and
uses the included tape.bin, so a compiler or Python is unnecessary.
The launcher uses the macOS scheduler.

The benchmark reports decision time for LANS, PPO, LQR and MPC on the
same 18,836 input states with three repeats. Median and P95 times are
in microseconds. host.txt records the system version, model identifier
and CPU brand. Use the same portable benchmark to compare CPU results.

Source: https://github.com/rizwanalihasalivi/LQR
