LANS timing benchmark for Windows x64

1. Extract the entire ZIP to a folder you can write to, such as Downloads.
2. Double-click Run-Benchmark.cmd in the extracted folder.
3. Wait for the result table. Send timing.csv and host.txt to compare results.

The ZIP includes the executable and benchmark data. The benchmark uses the
CPU and measures LANS, PPO, LQR and MPC on the same 18,836 input states.
The default run repeats the tape three times and reports median and P95
decision time in microseconds. Lower times are faster.

The run pins its thread to the first logical CPU available to the process.
On hybrid CPUs such as the Intel i5-1235U, host.txt records which logical
CPU was used. Keep power settings consistent across repeat measurements.

The executable uses the portable implementation from the legacy Intel
benchmark. Timing comparisons across machines should use the same bundle.
host.txt records the Windows version, CPU and chosen logical CPU. It does
not collect Windows device IDs or product IDs.

Source: https://github.com/rizwanalihasalivi/LQR
