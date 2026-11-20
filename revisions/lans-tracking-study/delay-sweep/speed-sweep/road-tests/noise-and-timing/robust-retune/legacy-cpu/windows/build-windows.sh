#!/bin/sh
set -eu
cd "$(dirname "$0")"
base=$(pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/lans-windows-build.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM
tar -xzf ../lans-legacy-intel.tar.gz -C "$work"
source="$work/lans-legacy-intel"
cp ../benchmark.cpp "$source/benchmark.cpp"
cc=${CC:-x86_64-w64-mingw32-gcc}
cxx=${CXX:-x86_64-w64-mingw32-g++}
mkdir -p "$work/build" "$work/package/LANS-Windows-x64"
package="$work/package/LANS-Windows-x64"
for file in "$source"/qp/src/*.c "$source"/qp/mpc90_workspace.c "$source"/qp/mpc150_workspace.c; do
    object="$work/build/$(basename "$file" .c).o"
    "$cc" -O3 -std=c99 -march=x86-64 -mno-avx \
        -I"$source/qp" -I"$source/qp/inc/public" -I"$source/qp/inc/private" -c "$file" -o "$object"
done
"$cxx" -O3 -std=c++17 -march=x86-64 -mno-avx -ffp-contract=off -static -static-libstdc++ -static-libgcc \
    -I"$source/qp" -I"$source/qp/inc/public" "$source/benchmark.cpp" "$work"/build/*.o -o "$package/LANS-Timing.exe"
cp "$source/tape.bin" "$source/OSQP-LICENSE" "$package/"
sed 's/$/\r/' Run-Benchmark.cmd > "$package/Run-Benchmark.cmd"
sed 's/$/\r/' README.txt > "$package/README.txt"
cp TOOLCHAIN-LICENSES.txt "$package/"
"$cxx" --version > "$package/compiler.txt"
(cd "$work/package" && zip -qr "$work/LANS-Windows-x64.zip" LANS-Windows-x64)
mv "$work/LANS-Windows-x64.zip" "$base/LANS-Windows-x64.zip"
