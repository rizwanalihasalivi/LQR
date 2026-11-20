#!/bin/sh
set -eu
cd "$(dirname "$0")"
{
    uname -srm
    case "$(uname -s)" in
        Darwin) sysctl -n machdep.cpu.brand_string ;;
        Linux) awk -F ': ' '/^model name/{print $2; exit}' /proc/cpuinfo ;;
    esac
    "${CXX:-c++}" --version
} > host.txt
./build/legacy-timing tape.bin "${1:-3}" > timing.csv
printf '%s\n' 'Hardware and compiler details are in host.txt. Timing results are in timing.csv.'
