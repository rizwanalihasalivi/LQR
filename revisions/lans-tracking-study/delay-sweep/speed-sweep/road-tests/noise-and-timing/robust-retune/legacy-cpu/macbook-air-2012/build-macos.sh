#!/bin/sh
set -eu
cd "$(dirname "$0")"
base=$(pwd)
case "${1:-}" in
    "") name=LANS-MacBook-Air-2012 ;;
    --energy) name=LANS-MacBook-Air-Energy ;;
    *) printf '%s\n' 'usage: build-macos.sh [--energy]' >&2; exit 1 ;;
esac
work=$(mktemp -d "${TMPDIR:-/tmp}/lans-macos-build.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM
tar -xzf ../lans-legacy-intel.tar.gz -C "$work"
source="$work/lans-legacy-intel"
cp ../benchmark.cpp "$source/benchmark.cpp"
zig=${ZIG:-zig}
target=x86_64-macos.10.13
export ZIG_GLOBAL_CACHE_DIR="${ZIG_GLOBAL_CACHE_DIR:-$work/global-cache}"
export ZIG_LOCAL_CACHE_DIR="${ZIG_LOCAL_CACHE_DIR:-$work/local-cache}"
package="$work/package/$name"
mkdir -p "$work/build" "$package"
for file in "$source"/qp/src/*.c "$source"/qp/mpc90_workspace.c "$source"/qp/mpc150_workspace.c; do
    object="$work/build/$(basename "$file" .c).o"
    "$zig" cc -target "$target" -O3 -std=c99 -march=x86_64 -mno-avx \
        -I"$source/qp" -I"$source/qp/inc/public" -I"$source/qp/inc/private" -c "$file" -o "$object"
done
"$zig" c++ -target "$target" -O3 -std=c++17 -march=x86_64 -mno-avx -ffp-contract=off \
    -I"$source/qp" -I"$source/qp/inc/public" "$source/benchmark.cpp" "$work"/build/*.o -o "$package/LANS-Timing"
cp "$source/tape.bin" "$source/OSQP-LICENSE" "$package/"
cp Run-Benchmark.command README.txt TOOLCHAIN-LICENSES.txt LLVM-RUNTIME-LICENSES.txt "$package/"
chmod +x "$package/Run-Benchmark.command" "$package/LANS-Timing"
if [ "${1:-}" = --energy ]; then
    cp Run-Energy.command ENERGY.txt "$package/"
    chmod +x "$package/Run-Energy.command"
fi
"$zig" version > "$package/compiler.txt"
(cd "$work/package" && zip -qr "$work/$name.zip" "$name")
mv "$work/$name.zip" "$base/$name.zip"
