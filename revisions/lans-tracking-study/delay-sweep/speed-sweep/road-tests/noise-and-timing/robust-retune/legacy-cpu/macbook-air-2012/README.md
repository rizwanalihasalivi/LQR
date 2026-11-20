# Intel MacBook Air (Mid 2012) benchmark

Extract `LANS-MacBook-Air-2012.zip` and double-click `Run-Benchmark.command`. The package includes the executable, the frozen benchmark tape and the solver. The launcher creates `timing.csv` and `host.txt` in the extracted folder. If starting it from Terminal, run `sh Run-Benchmark.command` from that folder.

The executable targets Intel x86-64 and macOS 10.13 High Sierra or later, including Catalina. It uses the same portable workload as the Windows bundle and measures LANS, PPO, LQR and MPC on the same inputs. macOS schedules the benchmark process.

The extracted package was verified as an x86-64 Mach-O executable with minimum macOS 10.13 and the standard system library. The launcher produced both output files for all four controllers in a Linux smoke test with a native stand-in executable. Actual macOS execution and MacBook timing remain to be verified on the device.

Rebuild on a Linux host with Zig 0.14.1 and ZIP installed by running `sh build-macos.sh`. The script reuses the [legacy Intel source bundle](../lans-legacy-intel.tar.gz) for the frozen input tape and solver workspaces.
