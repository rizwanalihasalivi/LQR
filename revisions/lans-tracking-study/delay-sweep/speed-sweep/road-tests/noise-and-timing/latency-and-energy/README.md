# Sensor-to-command distance and energy comparison

## Simulated sensor-to-steering apply distance

The replay uses the [matched Xeon CPU traces](../timing_decisions.csv) with 217,056 observed per-decision durations in 96 road, speed, delay and controller cells. These are the retained switching LANS, frozen PPO, LQR and objective-matched MPC timing traces. They cover Straight, Single Lane Change, Double Lane Change and Continuous Curve at 90 and 150 km/h and 0, 200 and 1000 ms fixed transport delay.

The model samples at the simulator's 100 Hz ticks. It runs each controller serially, including any backlog from a previous sample. It adds the scenario transport delay after computation and preserves command order. The first next eligible actuator tick applies the newest ready command. The replay drains scheduled commands after the final recorded sensor sample and counts any commands superseded at a shared tick. Distance is the specified constant speed multiplied by the elapsed time from the sensor sample. The CSVs include both the continuous command-ready time (before actuator polling) and the sampled command-apply time.

| Speed | Transport delay | P95 sensor-to-simulated-apply | P95 distance, each controller |
| --- | ---: | ---: | ---: |
| 90 km/h | 0 ms | 10 ms | 0.250 m |
| 90 km/h | 200 ms | 210 ms | 5.250 m |
| 90 km/h | 1000 ms | 1010 ms | 25.250 m |
| 150 km/h | 0 ms | 10 ms | 0.417 m |
| 150 km/h | 200 ms | 210 ms | 8.750 m |
| 150 km/h | 1000 ms | 1010 ms | 42.083 m |

The four controllers tie at sampled apply time in this replay. All recorded computation durations fit within a single 10 ms actuator period. None of the 217,056 scheduled commands were superseded. The distance before rounding to the actuator tick has controller-specific differences in the CSVs. The actuator boundary is the simulator's software command-apply event. A physical distance comparison needs synchronized sensor and steering-motion telemetry.

Reproduce with `python latency_replay.py`.

[Speed and delay results](distance_by_speed_delay.csv) · [Road-wise results](distance_by_cell.csv)

## Physical distance input

On an instrumented rig, timestamp each sensor sample and the first observed physical steering response to its command on a shared clock. Join the same command ID and log odometry at both events. Use repeatable conditions for all four controllers and retain missing or superseded commands in the capture record. The distance summary accepts matched applied events with these columns.

```text
controller,run_id,command_id,sensor_ns,actuator_ns,sensor_odometer_m,actuator_odometer_m
```

Run `python physical_distance.py EVENTS.csv` to calculate latency and odometer difference percentiles by controller. Physical measurement input is currently pending.

## Energy measurement

The cloud Xeon exposes neither RAPL powercap counters nor a package power event source. Hardware energy results are currently pending. The prepared workload uses the same four controllers and portable timing tape as the physical MacBook and Windows timing benchmarks. It warms and verifies the selected controller before the measurement window. Workload phases run complete tape sweeps and record actual elapsed time and decision count. The idle phase records a matched baseline.

### Intel MacBook power capture

Extract [the energy capture bundle](../robust-retune/legacy-cpu/macbook-air-2012/LANS-MacBook-Air-Energy.zip) and open `Run-Energy.command`. The launcher requests local permission to read the built-in `powermetrics` package power samples. It captures the idle phase and one workload window per controller. The resulting `energy-capture.*` folder contains raw power samples, workload windows, command counts and diagnostic logs. The full capture takes about six minutes.

The analysis command is `python energy_report.py --mac CAPTURE_FOLDER`. It integrates the sampled package power over each workload window and divides by the recorded decision count. It includes the range caused by whole-second timestamp rounding and an optional idle-adjusted column. It rejects unbracketed workload windows and mixed or missing package-power sources. If macOS labels its package power as model-derived, the result remains an OS-derived package energy estimate. For external metering, align the measured joule window to the recorded workload timestamps.

### Linux hardware package counter

Extract the [legacy source bundle](../robust-retune/legacy-cpu/lans-legacy-intel.tar.gz), copy the repository's updated [benchmark.cpp](../robust-retune/legacy-cpu/benchmark.cpp) over the extracted source, and build with `sh build.sh`. On a Linux machine with a readable package domain, use the same package counter across phases:

```sh
for controller in IDLE LANS PPO LQR MPC; do
    ./build/legacy-timing --workload tape.bin "$controller" 35 \
        /sys/class/powercap/intel-rapl:0/energy_uj > "$controller.csv"
done
```

Run `python energy_report.py --rapl IDLE.csv LANS.csv PPO.csv LQR.csv MPC.csv` against the captured rows. The runner reads start, periodic and end counter samples and handles a wrap between consecutive readings. The summary reports raw package joules per decision and the idle-adjusted estimate. The package includes processor activity outside the measured process. Repeat phases on the same hardware, counter and power settings for comparisons.

## Verification

The replay checked all source timing rows. Synthetic checks exercised FIFO supersession, energy-counter wrap, macOS text-sample integration and physical odometer aggregation. The portable workload ran all five phases on the Linux cloud host without a counter, with the joule fields left empty. The Mac bundle was cross-built for Intel macOS 10.13 or later. Physical actuator timing and Mac power collection require the actual device.
