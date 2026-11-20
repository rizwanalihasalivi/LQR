# Windows x64 benchmark

Download and extract `LANS-Windows-x64.zip`, then double-click `Run-Benchmark.cmd`. The executable and timing tape are included. The launcher creates `timing.csv` and `host.txt` in the extracted folder. All four controllers receive the same input tape on the first logical CPU available to the process.

The ZIP was built with MinGW-w64 GCC 14. The executable imports only the Windows system libraries `KERNEL32.dll` and `msvcrt.dll`. A fresh extraction completed the bundled launcher under Wine 10 with 56,508 decisions for each controller and both output files. Physical CPU results come from the user's Windows run.

To rebuild on a Linux host, run `sh build-windows.sh` with MinGW-w64 and ZIP installed. The script reuses the [legacy Intel source bundle](../lans-legacy-intel.tar.gz) for the frozen input tape and OSQP workspaces.
