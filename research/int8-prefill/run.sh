#!/bin/sh
# Build and run the W4A8 prefill microbenchmark.
set -e
cd "$(dirname "$0")"
mkdir -p out
xcrun -sdk macosx metal -std=metal4.0 -O3 -mmacosx-version-min=26.4 -c w4a8_bench.metal -o out/w4a8_bench.air
xcrun -sdk macosx metallib out/w4a8_bench.air -o out/w4a8_bench.metallib
clang++ -std=c++20 -O2 -fobjc-arc -mmacosx-version-min=26.4 w4a8_bench.mm -o out/w4a8_bench \
  -framework Foundation -framework Metal
./out/w4a8_bench out/w4a8_bench.metallib "$@"
