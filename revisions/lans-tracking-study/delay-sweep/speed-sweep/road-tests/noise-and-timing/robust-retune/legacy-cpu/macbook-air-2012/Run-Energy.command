#!/bin/sh
set -eu
cd "$(dirname "$0")"
meter=/usr/bin/powermetrics
if [ ! -x "$meter" ]; then
    printf '%s\n' 'This capture needs the built-in macOS powermetrics utility.' >&2
    exit 1
fi
result=$(mktemp -d "$PWD/energy-capture.XXXXXX")
{
    sw_vers
    uname -m
    sysctl -n hw.model
    sysctl -n machdep.cpu.brand_string
} > "$result/host.txt"
printf '%s\n' 'macOS asks for your password to read its CPU package power samples.' \
    'Keep this window open for about six minutes.'
sudo -v
meter_pid=
trap 'if [ -n "$meter_pid" ]; then kill "$meter_pid" 2>/dev/null || :; fi' EXIT
trap 'exit 130' INT TERM HUP
for name in IDLE LANS PPO LQR MPC; do
    printf 'Recording %s\n' "$name"
    sudo -n /usr/bin/env LC_ALL=C TZ=UTC "$meter" --samplers cpu_power -i 1000 -n 65 \
        > "$result/$name-power.txt" 2> "$result/$name-power-errors.txt" &
    meter_pid=$!
    sleep 3
    if ! kill -0 "$meter_pid" 2>/dev/null; then
        cat "$result/$name-power-errors.txt" >&2
        printf 'Capture folder: %s\n' "$result" >&2
        exit 1
    fi
    if ! ./LANS-Timing --workload tape.bin "$name" 35 > "$result/$name-workload.csv" 2> "$result/$name-workload-errors.txt"; then
        cat "$result/$name-workload-errors.txt" >&2
        exit 1
    fi
    if ! wait "$meter_pid"; then
        meter_pid=
        cat "$result/$name-power-errors.txt" >&2
        exit 1
    fi
    meter_pid=
done
printf '\n%s\n%s\n' 'Done. Share the files from this folder for the energy comparison.' "$result"
