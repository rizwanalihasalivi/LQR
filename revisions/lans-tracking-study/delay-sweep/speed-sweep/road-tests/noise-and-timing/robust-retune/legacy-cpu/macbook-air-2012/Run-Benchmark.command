#!/bin/sh
set -eu
cd "$(dirname "$0")"
printf '%s\n' 'Running LANS, PPO, LQR and MPC on this Intel Mac.' 'Leave this window open until the result table appears.'
{
    sw_vers
    uname -m
    sysctl -n hw.model
    sysctl -n machdep.cpu.brand_string
} > host-tmp.txt
if ./LANS-Timing tape.bin 3 > timing-tmp.csv 2>> host-tmp.txt; then
    mv timing-tmp.csv timing.csv
    mv host-tmp.txt host.txt
    cat timing.csv
    printf '\n%s\n' 'Done. Send timing.csv and host.txt from this folder for the comparison.'
else
    printf '\n%s\n' 'Benchmark failed. Details are shown below.'
    cat host-tmp.txt
    exit 1
fi
