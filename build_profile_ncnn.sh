#!/bin/bash
# Build profile_ncnn.cpp (the C++ counterpart to profile_ncnn.py -- see that
# file's docstring and profile_ncnn.cpp's header comment for why it exists).
#
# Usage: ./build_profile_ncnn.sh
# Produces ./profile_ncnn, linked against the ncnn build checked into
# ncnn-src/install/.
set -e

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ncnn_dir="$here/ncnn-src/install"

if [ ! -f "$ncnn_dir/lib/libncnn.a" ]; then
    echo "error: $ncnn_dir/lib/libncnn.a not found -- build/install ncnn-src first" >&2
    exit 1
fi

g++ -O2 -std=c++17 -pthread -fopenmp \
    -I "$ncnn_dir/include" \
    -o "$here/profile_ncnn" "$here/profile_ncnn.cpp" \
    -L "$ncnn_dir/lib" -lncnn

echo "Built $here/profile_ncnn"
