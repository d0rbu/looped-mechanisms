#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "usage: native/build.sh LLAMA_CPP_SOURCE_DIR LLAMA_LIBRARY_DIR" >&2
    exit 2
fi

source_dir=$(realpath "$1")
library_dir=$(realpath "$2")
if [[ ! -f "$library_dir/libllama.so" ]]; then
    echo "libllama.so is missing from $library_dir" >&2
    exit 2
fi

root=$(cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$root/build"
c++ -std=c++20 -O2 -Wall -Wextra -Werror -shared -fPIC \
    -I"$source_dir/include" -I"$source_dir/ggml/include" \
    "$root/native/capture.cpp" -L"$library_dir" -llama -lggml -lggml-base \
    -Wl,-rpath,"$library_dir" -o "$root/build/libloop_capture.so"
