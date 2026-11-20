#!/bin/sh
set -eu
cd "$(dirname "$0")"
arch=$(uname -m)
case "$arch" in
    x86_64|amd64) ;;
    *) printf '%s\n' 'This bundle targets 64-bit Intel-compatible hosts.' >&2; exit 1 ;;
esac
cc=${CC:-cc}
cxx=${CXX:-c++}
mkdir -p build
for source in qp/src/*.c qp/mpc90_workspace.c qp/mpc150_workspace.c; do
    obj="build/$(basename "$source" .c).o"
    "$cc" -O3 -std=c99 -march=x86-64 -mno-avx -Iqp -Iqp/inc/public -Iqp/inc/private -c "$source" -o "$obj"
done
"$cxx" -O3 -std=c++17 -march=x86-64 -mno-avx -ffp-contract=off -Iqp -Iqp/inc/public \
    benchmark.cpp build/*.o -o build/legacy-timing
